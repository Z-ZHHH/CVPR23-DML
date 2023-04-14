import numpy as np
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from common.ood_tools import get_ood_gradnorm
#from models.utils import build_model
from datasets.utils import build_dataset
from common.ood_tools import get_ood_scores
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn
from models.wrn import WideResNet
from models.resnet50 import *
from models.resnet34 import *
from models.densenet import *
from torchvision.models import densenet121



if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from common.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import common.score_calculation as lib
parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=5, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
#parser.add_argument('--method_name', '-s', type=str, default='cifar10_clean_00_allconv_standard', help='Method name.')

parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet'],
                    help='Choose between CIFAR-10, CIFAR-100, ImageNet.')
# Loading details
parser.add_argument('--net', default='wrn', type=str, help='wrn or resnet50')
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--gpu', type=str, default="0", help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=8, help='Pre-fetching threads.')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')

# DML 
parser.add_argument('--MCF', type=str, default="0", help='MCF model path')
parser.add_argument('--MNC', type=str, default="0", help='MCF model path')

parser.add_argument('--include_train', '-t', action='store_true',
                    help='test model on train set')
parser.add_argument('--optimal_t', action='store_true',
                    help='test model on train set')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = True  # fire on all cylinders



if args.gpu is not None:
    if len(args.gpu) == 1:
        device = torch.device('cuda:{}'.format(int(args.gpu)))
    else:
        device = torch.device('cuda:0')
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device('cpu')


def get_ood_scores(args, net_MCF, net_MNC, loader, ood_num_examples, device, in_dist=False):
    _score1 = []   
    _score2 = []   
    _right_score = []
    _wrong_score = []
    targets_all = []

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.to(device)
            output1, features1 = net_MCF(data)
            output2, features2 = net_MNC(data)
 
            all_score1 = np.max(to_np(output1), axis=1)    #focal cosine  attention norm=40           
            all_score2 = to_np(features2.norm(2, dim=1))     #center norm

            _score1.append(all_score1)
            _score2.append(all_score2)
    _score1 = concat(_score1)
    _score2 = concat(_score2)

    if in_dist:
        return _score1.copy(), _score2.copy()
    else:
        return _score1[:ood_num_examples].copy(), _score2[:ood_num_examples].copy()


def build_model(num_classes, device, args, use_norm, feature_norm):
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate, use_norm=use_norm, feature_norm=feature_norm)
    #net = create_model(num_classes=num_classes, use_norm=use_norm, feature_norm=feature_norm)
    net.to(device)
    if args.gpu is not None and len(args.gpu) > 1:
        gpu_list = [int(s) for s in args.gpu.split(',')]
        net = torch.nn.DataParallel(net, device_ids=gpu_list)
    return net
        
        
test_data, num_classes = build_dataset(args.dataset, mode="test")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)


# /////////////// Load MCF and MNC ///////////////

# Create model
net_MCF = build_model(num_classes, device, args, True, True)
net_MNC = build_model(num_classes, device, args, True, False)



path = './ckpt'
assert args.MCF!="0", 'MCF path is empty'
assert args.MNC!="0", 'MNC path is empty'

model_name = os.path.join(path, args.MCF)
if os.path.isfile(model_name):
    print('loading models')
    weights = torch.load(model_name, map_location=device)
    weights = {
        k: weights[k]
        if k in weights
        else net_MCF.state_dict()[k]
        for k in net_MCF.state_dict()
    } 

    net_MCF.load_state_dict(weights)

model_name = os.path.join(path, args.MNC)
if os.path.isfile(model_name):
    print('loading models')
    weights = torch.load(model_name, map_location=device)
    weights = {
        k: weights[k]
        if k in weights
        else net_MNC.state_dict()[k]
        for k in net_MNC.state_dict()
    } 

    net_MNC.load_state_dict(weights)
    
net_MCF.eval()
net_MNC.eval()
net_MCF.to(device)
net_MNC.to(device)
cudnn.benchmark = True  # fire on all cylinders



# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))


in_score1, in_score2 = get_ood_scores(args, net_MCF, net_MNC, test_loader, ood_num_examples, device, in_dist=True)



# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, in_score1, in_score2, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):

        out_score1, out_score2 = get_ood_scores(args, net_MCF, net_MNC, ood_loader, ood_num_examples, device)

        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
        
            ##############################
            tmp1 = np.sum(in_score1)
            tmp2 = np.sum(out_score1)

            in_score1_tmp = in_score1/tmp1
            out_score1_tmp = out_score1/tmp1
            
            tmp1 = np.sum(in_score2)
            tmp2 = np.sum(out_score2)
            in_score2_tmp = in_score2/tmp1
            out_score2_tmp = out_score2/tmp1
            

            in_score = in_score1_tmp + in_score2_tmp  
            out_score = out_score1_tmp + out_score2_tmp

            ############################
            measures = get_measures(in_score, out_score)
            
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs)
    else:
        print_measures(auroc, aupr, fpr)
    return out_score


if __name__ == '__main__':
    OOD_data_list = ["Textures", "SVHN", "LSUN-C", "LSUN-R", "iSUN", "Places365"]

    for data_name in OOD_data_list:
        if data_name == args.dataset:
            continue
        ood_data, _ = build_dataset(data_name, mode="test")
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                     num_workers=args.prefetch, pin_memory=True)
        print('\n\n{} Detection'.format(data_name))
        out_score = get_and_print_results(ood_loader,in_score1, in_score2)

    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list))







