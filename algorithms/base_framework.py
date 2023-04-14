from tqdm import tqdm
import abc, os
from models.utils import build_model
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np

      
        
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class SingleModel:
    __metaclass__ = abc.ABCMeta
    def __init__(self, args, device, num_classes):
        self.device = device
        self.args = args
        self.num_classes = num_classes
        # Create model
        self.net = build_model(args.model, num_classes, device, args)
        
        self.iterations = 0

        self.optimizer_model = torch.optim.SGD(self.net.parameters(), args.learning_rate,
                                                momentum=args.momentum, weight_decay=args.decay)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_model, args.epochs, eta_min=0.0)

        if args.loss == "ce":
            self.loss_function = torch.nn.CrossEntropyLoss()
            
        elif args.loss == "logit_norm":
            from common.loss_function import LogitNormLoss
            self.loss_function = LogitNormLoss(device, self.args.temp)
            
        elif args.loss == "centerloss":
            from common.CenterLoss import CenterLoss
            self.loss_function = CenterLoss(device)
            self.center_optimizer = torch.optim.SGD(self.loss_function.parameters(), lr=0.5)

        elif args.loss == "Focal":
            from common.FocalLoss import FocalLoss
            self.loss_function = FocalLoss(device)

    def train(self, train_loader, epoch): # one epoch
        self.net.train()
        loss_avg = 0.0
        
        if self.args.loss == "centerloss":

            center_milestones = [0, 60,80]
            assigned_center_weights = [0.0, 0.001,0.005]
            center_weight = assigned_center_weights[0]
            for i, ms in enumerate(center_milestones):
                if epoch >= ms:
                    center_weight = assigned_center_weights[i]
            

        for data, target, index in tqdm(train_loader):

            index = 0
            self.optimizer_model.zero_grad()
            if self.args.loss == "centerloss":
                self.center_optimizer.zero_grad()
                loss_ct, loss_ce = self.train_batch(index, data, target, epoch)
                loss = loss_ce + loss_ct*center_weight
            else:
                loss = self.train_batch(index, data, target, epoch)
            
            loss.backward()
            self.optimizer_model.step()
            
            if self.args.loss == "centerloss":
                # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                for param in self.loss_function.parameters():
                    param.grad.data *= (1./(center_weight + 1e-12))
                self.center_optimizer.step()
                
            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        self.scheduler.step()
        return loss_avg


    @abc.abstractmethod
    def train_batch(self, batch_idx, inputs, targets, epoch):
        pass

    def test(self, test_loader):
        self.net.eval()
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for dict in test_loader:
                data, target = dict[0].to(self.device), dict[1].to(self.device)

                # forward
                output,featrues = self.net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        return loss_avg / len(test_loader), correct / len(test_loader.dataset)
