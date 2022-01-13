import gc

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, resnet101, resnet18


class EdgeNet(nn.Module):

    def __init__(self, base_model=resnet18, num_templates=1, num_objects=1, enable_edge=False):
        super().__init__()
        self.model = base_model(pretrained=True)
        self.num_templates = num_templates
