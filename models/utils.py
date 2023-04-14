from models.wrn import WideResNet
from models.resnet50 import *
from models.resnet34 import *
from models.densenet import *
import torch
from torchvision.models import densenet121
import numpy as np

def build_model(model_type, num_classes, device, args):
    if args.net == 'resnet50':
        print('Using ResNet50')
        net = resnet50(pretrained=False, use_norm=False, feature_norm=False)  #True
    elif args.net == 'resnet34':
        print('Using ResNet34')
        net = create_model(num_classes=num_classes, use_norm=False, feature_norm=False)  #True
    elif args.net == 'densenet':
        print('Using DenseNet')
        net = DenseNet(num_classes=num_classes,  use_norm=True, feature_norm=False)  #True
    else:
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    net.to(device)
    if args.gpu is not None and len(args.gpu) > 1:
        gpu_list = [int(s) for s in args.gpu.split(',')]
        net = torch.nn.DataParallel(net, device_ids=gpu_list)
    return net
