# import gc
#
# import numpy as np
# import torch
# from torch import nn
# from torchvision.models import resnet50, resnet101, vgg16, resnet18
#
# from models.EdgeNet import EdgeNet
#
#
# class DetectionModel(nn.Module):
#     """
#     Hybrid Model from Tiny Faces paper
#     """
#
#     def __init__(self, base_model=resnet50, edge_model=vgg16, num_templates=1, num_objects=1, enable_edge=False):
#         super().__init__()
#         # 4 is for the bounding box offsets
#         output = (num_objects + 4)*num_templates
#         self.num_templates = num_templates
#         self.model = base_model(pretrained=True)
#         self.edge_model = EdgeNet(num_templates=num_templates)
#         self.enable_edge = enable_edge
#
#         if enable_edge:
#             # del self.edge_model.layer4
#             # self.edge_model.conv1.in_channels = 1
#             # weights = torch.nn.init.kaiming_normal_(torch.nn.Parameter(torch.FloatTensor(64, 1, 7, 7)), a=0, mode='fan_in', nonlinearity='leaky_relu')
#             # self.edge_model.conv1.weight = weights
#             self.score_res5 = nn.Conv2d(in_channels=256, out_channels=num_templates, kernel_size=1, padding=0)
#
#         # delete unneeded layer
#         del self.model.layer4
#         self.score_res3 = nn.Conv2d(in_channels=512, out_channels=output,
#                                     kernel_size=1, padding=0)
#         self.score_res4 = nn.Conv2d(in_channels=1024, out_channels=output,
#                                     kernel_size=1, padding=0)
#
#         self.score4_upsample = nn.ConvTranspose2d(in_channels=output, out_channels=output,
#                                                   kernel_size=4, stride=2, padding=1, bias=False)
#         self._init_bilinear()
#
#     def _init_weights(self):
#         pass
#
#     def _init_bilinear(self):
#         """
#         Initialize the ConvTranspose2d layer with a bilinear interpolation mapping
#         :return:
#         """
#         k = self.score4_upsample.kernel_size[0]
#         factor = np.floor((k+1)/2)
#         if k % 2 == 1:
#             center = factor
#         else:
#             center = factor + 0.5
#         C = np.arange(1, 5)
#
#         f = np.zeros((self.score4_upsample.in_channels,
#                       self.score4_upsample.out_channels, k, k))
#
#         for i in range(self.score4_upsample.out_channels):
#             f[i, i, :, :] = (np.ones((1, k)) - (np.abs(C-center)/factor)).T @ \
#                             (np.ones((1, k)) - (np.abs(C-center)/factor))
#
#         self.score4_upsample.weight = torch.nn.Parameter(data=torch.Tensor(f))
#
#     def learnable_parameters(self, lr):
#         parameters = [
#             # Be T'Challa. Don't freeze.
#             {'params': self.model.parameters(), 'lr': lr},
#             {'params': self.score_res3.parameters(), 'lr': 0.1*lr},
#             {'params': self.score_res4.parameters(), 'lr': 1*lr},
#             {'params': self.score4_upsample.parameters(), 'lr': 0}  # freeze UpConv layer
#         ]
#         return parameters
#
#     def forward(self, x):
#         if self.enable_edge:
#             # x2 = self.edge_model.conv1(x[:, -2: -1, :, :])
#             # x2 = self.edge_model.bn1(x2)
#             # x2 = self.edge_model.relu(x2)
#             # x2 = self.edge_model.maxpool(x2)
#             # x2 = self.edge_model.layer1(x2)
#             # x2 = self.edge_model.layer2(x2)
#             x2 = self.edge_model(x[:, -2: -1, :, :])
#             score_res5 = self.score_res5(x2)
#         x = self.model.conv1(x[:, :-1, :, :]) # 1 64 250 250
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x) # 1 64 125 125
#
#         x = self.model.layer1(x) # 1 256 125 125
#         # res2 = x
#
#         x = self.model.layer2(x) # 1 512 63 63
#         res3 = x
#
#         x = self.model.layer3(x) # 1 1024 32 32
#         res4 = x
#
#         score_res3 = self.score_res3(res3) # 1 125 63 63
#
#         score_res4 = self.score_res4(res4) # 1 125 32 32
#         score4 = self.score4_upsample(score_res4) # 1 125 64 64
#
#         # We need to do some fancy cropping to accomodate the difference in image sizes in eval
#         if not self.training:
#             # from vl_feats DagNN Crop
#             cropv = score4.size(2) - score_res3.size(2)
#             cropu = score4.size(3) - score_res3.size(3)
#             # if the crop is 0 (both the input sizes are the same)
#             # we do some arithmetic to allow python to index correctly
#             if cropv == 0:
#                 cropv = -score4.size(2)
#             if cropu == 0:
#                 cropu = -score4.size(3)
#
#             score4 = score4[:, :, 0:-cropv, 0:-cropu]
#         else:
#             # match the dimensions arbitrarily
#             score4 = score4[:, :, 0:score_res3.size(2), 0:score_res3.size(3)]
#         if not self.enable_edge:
#             score = score_res3 + score4
#         else:
#             score:torch.Tensor = score_res3[:, :self.num_templates, :, :] + score4[:, :self.num_templates, :, :] + score_res5
#             score_reg = score_res3[:, self.num_templates:, :, :] + score4[:, self.num_templates:, :, :]
#             score = torch.cat((score, score_reg), 1)
#         gc.collect()
#         torch.cuda.empty_cache()
#         return score
import gc

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, resnet101, vgg16, resnet18


class DetectionModel(nn.Module):
    """
    Hybrid Model from Tiny Faces paper
    """

    def __init__(self, base_model=resnet50, edge_model=resnet18, num_templates=1, num_objects=1, enable_edge=False):
        super().__init__()
        # 4 is for the bounding box offsets
        output = (num_objects + 4)*num_templates
        self.num_templates = num_templates
        self.model = base_model(pretrained=True)

        self.enable_edge = enable_edge
        if enable_edge:
            # self.model.conv1.in_channels = 4
            # weight1 = torch.nn.init.kaiming_normal_(torch.nn.Parameter(torch.FloatTensor(64, 4, 7, 7)), a=0, mode='fan_in', nonlinearity='leaky_relu')
            # self.model.conv1.weight = weight1
            self.edge_model = edge_model(pretrained=True)
            del self.edge_model.layer4
            self.edge_model.conv1.in_channels = 1
            weights = torch.nn.init.kaiming_normal_(torch.nn.Parameter(torch.FloatTensor(64, 1, 7, 7)), a=0, mode='fan_in', nonlinearity='leaky_relu')
            self.edge_model.conv1.weight = weights
            self.score_res5 = nn.Conv2d(in_channels=128, out_channels=num_templates, kernel_size=1, padding=0)

        # delete unneeded layer
        del self.model.layer4
        self.score_res3 = nn.Conv2d(in_channels=512, out_channels=output,
                                    kernel_size=1, padding=0)
        self.score_res4 = nn.Conv2d(in_channels=1024, out_channels=output,
                                    kernel_size=1, padding=0)

        self.score4_upsample = nn.ConvTranspose2d(in_channels=output, out_channels=output,
                                                  kernel_size=4, stride=2, padding=1, bias=False)
        self._init_bilinear()

    def _init_weights(self):
        pass

    def _init_bilinear(self):
        """
        Initialize the ConvTranspose2d layer with a bilinear interpolation mapping
        :return:
        """
        k = self.score4_upsample.kernel_size[0]
        factor = np.floor((k+1)/2)
        if k % 2 == 1:
            center = factor
        else:
            center = factor + 0.5
        C = np.arange(1, 5)

        f = np.zeros((self.score4_upsample.in_channels,
                      self.score4_upsample.out_channels, k, k))

        for i in range(self.score4_upsample.out_channels):
            f[i, i, :, :] = (np.ones((1, k)) - (np.abs(C-center)/factor)).T @ \
                            (np.ones((1, k)) - (np.abs(C-center)/factor))

        self.score4_upsample.weight = torch.nn.Parameter(data=torch.Tensor(f))

    def learnable_parameters(self, lr):
        parameters = [
            # Be T'Challa. Don't freeze.
            {'params': self.model.parameters(), 'lr': lr},
            {'params': self.score_res3.parameters(), 'lr': 0.1*lr},
            {'params': self.score_res4.parameters(), 'lr': 1*lr},
            {'params': self.score4_upsample.parameters(), 'lr': 0}  # freeze UpConv layer
        ]
        return parameters

    def forward(self, x):
        if self.enable_edge:
            x2 = self.edge_model.conv1(x[:, -2: -1, :, :])
            x2 = self.edge_model.bn1(x2)
            x2 = self.edge_model.relu(x2)
            x2 = self.edge_model.maxpool(x2)
            x2 = self.edge_model.layer1(x2)
            x2 = self.edge_model.layer2(x2)
            score_res5 = self.score_res5(x2)
            x = self.model.conv1(x[:, :-1, :, :])  # 1 64 250 250
        else:
            x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) # 1 64 125 125

        x = self.model.layer1(x) # 1 256 125 125
        # res2 = x

        x = self.model.layer2(x) # 1 512 63 63
        res3 = x

        x = self.model.layer3(x) # 1 1024 32 32
        res4 = x

        score_res3 = self.score_res3(res3) # 1 125 63 63

        score_res4 = self.score_res4(res4) # 1 125 32 32
        score4 = self.score4_upsample(score_res4) # 1 125 64 64

        # We need to do some fancy cropping to accomodate the difference in image sizes in eval
        if not self.training:
            # from vl_feats DagNN Crop
            cropv = score4.size(2) - score_res3.size(2)
            cropu = score4.size(3) - score_res3.size(3)
            # if the crop is 0 (both the input sizes are the same)
            # we do some arithmetic to allow python to index correctly
            if cropv == 0:
                cropv = -score4.size(2)
            if cropu == 0:
                cropu = -score4.size(3)

            score4 = score4[:, :, 0:-cropv, 0:-cropu]
        else:
            # match the dimensions arbitrarily
            score4 = score4[:, :, 0:score_res3.size(2), 0:score_res3.size(3)]
        if not self.enable_edge:
            score = score_res3 + score4
        else:
            score:torch.Tensor = score_res3[:, :self.num_templates, :, :] + score4[:, :self.num_templates, :, :] + score_res5
            score_reg = score_res3[:, self.num_templates:, :, :] + score4[:, self.num_templates:, :, :]
            score = torch.cat((score, score_reg), 1)
        gc.collect()
        torch.cuda.empty_cache()
        return score
