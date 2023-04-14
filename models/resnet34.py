import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers.ModulatedAttLayer import ModulatedAttLayer
import os, random, time, copy, scipy, pickle, sys, math
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import Parameter

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        #out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        out = x.mm(F.normalize(self.weight, dim=0))
        return out
        
        
class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, num_classes=100, isPretrained=False, isGrayscale=False, poolSize=4, use_norm=True, feature_norm=False):
        super(ResnetEncoder, self).__init__()
        self.path_to_model = '/tmp/models'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.isGrayscale = isGrayscale
        self.isPretrained = isPretrained

        self.poolSize = poolSize
        self.featListName = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        
        self.use_norm=use_norm
        self.feature_norm=feature_norm
        
        resnets = {
            18: models.resnet18, 
            34: models.resnet34,
            50: models.resnet50, 
            101: models.resnet101,
            152: models.resnet152}
        
        resnets_pretrained_path = {
            18: 'resnet18-5c106cde.pth', 
            34: 'resnet34.pth',
            50: 'resnet50-19c8e357.pth',
            101: 'resnet101.pth',
            152: 'resnet152.pth'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))

        self.encoder = resnets[num_layers]()
        
        if self.isPretrained:
            print("using pretrained model")
            self.encoder.load_state_dict(
                torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
            
        if self.isGrayscale:
            self.encoder.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        if num_layers > 34:
            self.num_ch_enc[1:] = 2048
        else:
            self.num_ch_enc[1:] = 512
            
        if self.use_norm:
            self.fc = NormedLinear(512, num_classes)
            print(self.fc.weight)
            print(self.fc.weight.shape)
        else:
            self.fc = nn.Linear(512, num_classes, bias=False)      
 
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()            
            

    def forward(self, input_image):
        self.features = []  
        
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        
        x = self.encoder.layer1(x)
        self.features.append(x)
        
        x = self.encoder.layer2(x)
        self.features.append(x)
        
        x = self.encoder.layer3(x) 
        self.features.append(x)
        
        x = self.encoder.layer4(x)
        self.features.append(x)
        
        x = F.avg_pool2d(x, self.poolSize)
        x = x.view(x.size(0), -1)

        if self.feature_norm:
            #print('F_norm')
            x = F.normalize(x, dim=1) * 40  #self.scale

        return self.fc(x), x
        
        
        

def create_model(num_classes=100, use_fc=True, dropout=None, use_norm=True, feature_norm=True):
    
    print('Loading Scratch ResNet 34 Feature Model.')
    resnet34 = ResnetEncoder(num_layers=34, num_classes=100, isPretrained=False, use_norm=use_norm, feature_norm=feature_norm)

    return resnet34
