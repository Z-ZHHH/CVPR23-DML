#!/usr/bin/env python
import argparse
import torch
import numpy as np
from tqdm import tqdm
#import mmcv
from numpy.linalg import norm, pinv
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp
import pandas as pd
from models.utils import build_model
from datasets.utils import build_dataset
import os
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn
from common.display_results import show_performance, get_measures, print_measures, print_measures_with_std


recall_level_default = 0.95

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    #parser.add_argument('fc', help='Path to config')
    #parser.add_argument('id_train_feature', help='Path to data')
    #parser.add_argument('id_val_feature', help='Path to output file')
    #parser.add_argument('ood_features', nargs="+", help='Path to ood features')
    #parser.add_argument('--train_label', default='datalists/imagenet2012_train_random_200k.txt', help='Path to train labels')
    parser.add_argument('--clip_quantile', default=0.99, help='Clip quantile to react')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
    parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
    parser.add_argument('--method_name', '-s', type=str, default='cifar10_clean_00_allconv_standard', help='Method name.')
    
    parser.add_argument('dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument('--score', default='MSP', type=str, help='score options: Odin|MSP|energy|gradnorm')
    # Loading details
    parser.add_argument('--net', default='wrn', type=str, help='wrn or resnet50')
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
    parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
    parser.add_argument('--gpu', type=str, default="1", help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
    parser.add_argument('--seed', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--T', default=1000., type=float, help='temperature: energy|Odin')
    parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
    parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
    
    parser.add_argument('--include_train', '-t', action='store_true',
                        help='test model on train set')
    parser.add_argument('--optimal_t', action='store_true',
                        help='test model on train set')

    return parser.parse_args()

    
#region Helper
def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out

def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

#endregion

def get_features(args, net, loader, ood_num_examples, device, in_dist=True):

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    
    feature_all = []
    label_all = []
    logits_all = []
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.to(device)
            output, features = net(data)
            logits_all.append(output)
            feature_all.append(features)
            label_all.append(target)
            
    feature_all = torch.vstack(feature_all)
    logits_all = torch.vstack(logits_all)
    label_all = torch.hstack(label_all)
    # ('feature_all shape', feature_all.shape)
    # print('logits_all shape', logits_all.shape)

    return feature_all, label_all, logits_all
    
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out
    
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

#region OOD

def gradnorm(x, w, b):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []
    ###############################
    for i in tqdm(x):
        targets = torch.ones((1, 100)).cuda()
        fc.zero_grad()
        loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

#endregion

def main():
    args = parse_args()
    #args.method_name = 'cifar10_wrn_CE_baseline_NN25_standard'
    #args.dataset = 'cifar10'

    if args.gpu is not None:
        if len(args.gpu) == 1:
            device = torch.device('cuda:{}'.format(int(args.gpu)))
            print(device)
        else:
            device = torch.device('cuda:0')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    #net.fc = nn.Linear(128, 10, bias=False)
    #net.to(device)
    
    train_data, num_classes = build_dataset(args.dataset, "train")
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.test_bs, shuffle=False,
            num_workers=args.prefetch, pin_memory=True)
    test_data, num_classes = build_dataset(args.dataset, mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
    ood_num_examples = len(test_data) // 5
    expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
    
    
    model_type = args.method_name.split("_", 5)[-3]
    alg = args.method_name.split("_", 5)[-1]
    net = build_model(model_type, num_classes, device, args)

    
    start_epoch = 0
    if args.load != '':
        for i in range(1000 - 1, -1, -1):
            model_name = os.path.join(os.path.join(args.load, alg), args.method_name + '_epoch_' + str(i) + '.pt')
            print(model_name)
            if os.path.isfile(model_name):
                weights = torch.load(model_name, map_location=device)
                weights = {
                    k: weights[k]
                    if k in weights
                    else net.state_dict()[k]
                    for k in net.state_dict()
                } 
                # loading
                weights['fc.weight'] = weights['fc.weight']
                net.load_state_dict(weights)
                w = np.array(weights['fc.weight'].clone().detach().cpu())  #.t()
                if 'fc.bias' in weights.keys():
                    b = np.array(weights['fc.bias'].clone().detach().cpu())
                    print('with bias')
                else:
                    b = np.array(torch.zeros(num_classes))

                print('Model restored! Epoch:', i)
                start_epoch = i + 1
                break
        if start_epoch == 0:
            assert False, "could not resume"
    start_epoch = 0
    net.eval()
    #cudnn.benchmark = True  # fire on all cylinders
    
    data_name = ["Textures", "SVHN", "LSUN-C", "LSUN-R", "iSUN", "Places365"]
    ood_names = data_name
    print(f"ood datasets: {data_name}")

    print('load features')
    feature_id_train,  train_labels, logit_id_train = get_features(args, net, train_loader, ood_num_examples, device, in_dist=True)
    feature_id_val, val_labels, logit_id_val = get_features(args, net, test_loader, ood_num_examples, device, in_dist=True)

    feature_oods_all = []
    logit_oods_all = []
    
    
    for i in range(10):
        logit_dict = {}
        feature_dict = {}
        for data in data_name:
            ood_data, _ = build_dataset(data, mode="test")
            ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                         num_workers=args.prefetch, pin_memory=True) 
            feature_oods, _, logit_oods = get_features(args, net, ood_loader, ood_num_examples, device, in_dist=False)
            feature_oods = np.array(feature_oods.cpu())
            logit_oods = np.array(logit_oods.cpu())
    
            logit_dict.update({data: logit_oods})
            feature_dict.update({data: feature_oods})
    
        logit_oods_all.append(logit_dict)
        feature_oods_all.append(feature_dict)
    
    train_labels = np.array(train_labels.cpu())
    val_labels = np.array(val_labels.cpu())
    logit_id_train = np.array(logit_id_train.cpu())
    logit_id_val = np.array(logit_id_val.cpu())

    feature_id_train = np.array(feature_id_train.cpu())
    feature_id_val = np.array(feature_id_val.cpu())

    recall = 0.95

    print('computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)

    softmax_oods = []
    for i in range(10):
        softmax_oods.append({name: softmax(logit, axis=-1) for name, logit in logit_oods_all[i].items()})   
    
    preds = np.argmax(softmax_id_val, axis=1)
    print(preds.shape)
    print(val_labels.shape)
    #tmp_list = [1]*preds.shape[0]
    tmp = preds[preds==val_labels]
    print(tmp)
    correct = len(tmp)
    #print(preds==val_labels)
    print('ID_ACC:', correct/len(test_loader.dataset))
    
    print(w,b, pinv(w))
    print(w.shape,b.shape, pinv(w).shape)
    u = -np.matmul(pinv(w), b)
    df = pd.DataFrame(columns=['method', 'oodset', 'auroc', 'fpr'])
    dfs = []

    # ---------------------------------------
    method = 'MSP'
    print(f'\n{method}')
    result = []
    score_id = softmax_id_val.max(axis=-1)
    auroc_list, aupr_list, fpr_list = [], [], []
    for name in softmax_oods[0].keys():
        print(name)
        aurocs, auprs, fprs = [], [], []
        for i in range(10):
            softmax_ood = softmax_oods[i][name]
            score_ood = softmax_ood.max(axis=-1)
            #auc_ood = auc(score_id, score_ood)[0]
            #fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            measures = get_measures(score_id[:], score_ood[:], 0.95)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        #result.append(dict(method=method, oodset=name, auroc=auroc, aupr=aupr, fpr=fpr95))
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

    
    # ---------------------------------------
    method = 'Odin'
    print(f'\n{method}')
    result = []
    Odin_id_val = softmax(logit_id_val/1000., axis=-1)
    score_id = Odin_id_val.max(axis=-1)
    Odin_oods = []
    auroc_list, aupr_list, fpr_list = [], [], []
    for i in range(10):
        Odin_oods.append({name: softmax(logit / 1000., axis=-1) for name, logit in logit_oods_all[i].items()})
    for name in softmax_oods[0].keys():
        aurocs, auprs, fprs = [], [], []
        print(name)
        for i in range(10):
            score_ood = Odin_oods[i][name].max(axis=-1)
            measures = get_measures(score_id[:], score_ood[:], 0.95)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        #result.append(dict(method=method, oodset=name, auroc=auroc, aupr=aupr, fpr=fpr95))
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

    # ---------------------------------------
    method = 'MaxLogit'
    print(f'\n{method}')
    result = []
    score_id = logit_id_val.max(axis=-1)
    auroc_list, aupr_list, fpr_list = [], [], []
    for name in softmax_oods[0].keys():
        aurocs, auprs, fprs = [], [], []
        print(name)
        for i in range(10):
            logit_ood_tmp = logit_oods_all[i][name]
            score_ood = logit_ood_tmp.max(axis=-1)
            measures = get_measures(score_id[:], score_ood[:], 0.95)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        #result.append(dict(method=method, oodset=name, auroc=auroc, aupr=aupr, fpr=fpr95))
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
    
    # ---------------------------------------
    method = 'Energy'
    print(f'\n{method}')
    result = []
    score_id = logsumexp(logit_id_val, axis=-1)
    auroc_list, aupr_list, fpr_list = [], [], []
    for name in softmax_oods[0].keys():
        aurocs, auprs, fprs = [], [], []
        print(name)
        for i in range(10):
            logit_ood_tmp = logit_oods_all[i][name]
            score_ood = score_ood = logsumexp(logit_ood_tmp, axis=-1)
            measures = get_measures(score_id[:], score_ood[:], 0.95)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        #result.append(dict(method=method, oodset=name, auroc=auroc, aupr=aupr, fpr=fpr95))
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

    # ---------------------------------------
    method = 'Energy+React'
    print(f'\n{method}')
    result = []

    clip = np.quantile(feature_id_train, args.clip_quantile)
    print(f'clip quantile {args.clip_quantile}, clip {clip:.4f}')
    logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
    score_id = logsumexp(logit_id_val_clip, axis=-1)
    auroc_list, aupr_list, fpr_list = [], [], []

    for name in softmax_oods[0].keys():
        aurocs, auprs, fprs = [], [], []
        print(name)
        for i in range(10):
            logit_ood_clip = np.clip(feature_oods_all[i][name], a_min=None, a_max=clip) @ w.T + b
            score_ood = logsumexp(logit_ood_clip, axis=-1)

            logit_ood_tmp = logit_oods_all[i][name]
            score_ood = score_ood = logsumexp(logit_ood_tmp, axis=-1)
            measures = get_measures(score_id[:], score_ood[:], 0.95)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        #result.append(dict(method=method, oodset=name, auroc=auroc, aupr=aupr, fpr=fpr95))
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

      
    # ---------------------------------------
    method = 'Mahalanobis'
    print(f'\n{method}')
    result = []

    print('computing classwise mean feature...')
    train_means = []
    train_feat_centered = []
    for i in tqdm(range(100)):
        fs = feature_id_train[train_labels == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    print('computing precision matrix...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

    print('go to gpu...')
    mean = torch.from_numpy(np.array(train_means)).cuda().float()
    prec = torch.from_numpy(ec.precision_).cuda().float()

    score_id = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(feature_id_val).cuda().float())])

    auroc_list, aupr_list, fpr_list = [], [], []

    for name in softmax_oods[0].keys():
        aurocs, auprs, fprs = [], [], []
        print(name)
        for i in range(5):
            score_ood = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(feature_oods_all[i][name]).cuda().float())])
            measures = get_measures(score_id[:], score_ood[:], 0.95)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        #result.append(dict(method=method, oodset=name, auroc=auroc, aupr=aupr, fpr=fpr95))
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

    #results.append(['Mah', np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list)])

    # ---------------------------------------
    method = 'GradNorm'
    print(f'\n{method}')
    result = []
    score_id = gradnorm(feature_id_val, w, b)
    auroc_list, aupr_list, fpr_list = [], [], []
    
    for name in softmax_oods[0].keys():
        aurocs, auprs, fprs = [], [], []
        print(name)
        for i in range(10):
            score_ood = gradnorm(feature_oods_all[i][name], w, b)
            measures = get_measures(score_id[:], score_ood[:], 0.95)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        #result.append(dict(method=method, oodset=name, auroc=auroc, aupr=aupr, fpr=fpr95))
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

    
    # ---------------------------------------
    method = 'ViM'
    print(f'\n{method}')
    result = []
    #DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
    DIM = 64  # we set the dim=64 because the WRN net feature dim is 128
    print(f'{DIM=}')

    print('computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    #print(eigen_vectors.shape)
    #print(eig_vals, eigen_vectors)
    #print(eigen_vectors.T[np.argsort(eig_vals * -1)].shape)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    print('computing alpha...')
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)

    #print(vlogit_id_train.mean())
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')

    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val
    
    auroc_list, aupr_list, fpr_list = [], [], []

    for name in softmax_oods[0].keys():
        aurocs, auprs, fprs = [], [], []
        print(name)
        for i in range(10):
            energy_ood = logsumexp(logit_oods_all[i][name], axis=-1)
            vlogit_ood = norm(np.matmul(feature_oods_all[i][name] - u, NS), axis=-1) * alpha
            score_ood = -vlogit_ood + energy_ood
            measures = get_measures(score_id[:], score_ood[:], 0.95)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        #result.append(dict(method=method, oodset=name, auroc=auroc, aupr=aupr, fpr=fpr95))
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
    

if __name__ == '__main__':
    main()
