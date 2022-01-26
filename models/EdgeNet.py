import gc
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, resnet101, resnet18, vgg16


class EdgeNet(nn.Module):

    def __init__(self, base_model=vgg16, num_templates=1, num_objects=1, enable_edge=False):
        super().__init__()
        self.model = base_model(pretrained=True)
        self.num_templates = num_templates
        self.features = nn.Sequential(OrderedDict([
            ('0', nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('0_bn', nn.BatchNorm2d(64)),
            ('1', nn.ReLU(inplace=True)),
            ('2', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('2_bn', nn.BatchNorm2d(64)),
            ('3', nn.ReLU(inplace=True)),
            ('4', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            ('5', nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('5_bn', nn.BatchNorm2d(128)),
            ('6', nn.ReLU(inplace=True)),
            ('7', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('7_bn', nn.BatchNorm2d(128)),
            ('8', nn.ReLU(inplace=True)),
            ('9', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            ('10', nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('10_bn', nn.BatchNorm2d(256)),
            ('11', nn.ReLU(inplace=True)),
            ('12', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('12_bn', nn.BatchNorm2d(256)),
            ('13', nn.ReLU(inplace=True)),
            ('14', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('14_bn', nn.BatchNorm2d(256)),
            ('15', nn.ReLU(inplace=True)),
            ('16', nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False))
        ]))
        torch.nn.init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        state_dict = base_model(pretrained=True).state_dict()
        state_dict.pop('features.0.weight')
        state_dict.pop('features.0.bias')
        state_dict.pop('features.17.weight')
        state_dict.pop('features.17.bias')
        self.features.load_state_dict(state_dict, False)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.features(x)
