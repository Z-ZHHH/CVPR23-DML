from algorithms.base_framework import SingleModel
import torch.nn.functional as F
import torch.nn as nn
import torch

class Standard(SingleModel):
    def train_batch(self, index, inputs, targets, epoch):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits, features = self.net(inputs)
        if self.args.loss == "simprototype":
            loss = self.loss_function(logits, features, targets)

        elif self.args.loss == "centerloss":
            loss_ct = self.loss_function(features, targets)
            loss_func = nn.CrossEntropyLoss()
            loss_ce = loss_func(logits, targets)
            return loss_ct, loss_ce
        else:
            #logitnorm
            #norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
            #logit_norm = torch.div(logits, norms) / 0.04
            #loss = self.loss_function(logit_norm, targets)
            
            
            loss = self.loss_function(logits, targets)
            return loss

