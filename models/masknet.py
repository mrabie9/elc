from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# from models.bayes_resnet import BayesianLinear
from dst import Dempster_Shafer_module, DM#



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MaskConv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = MaskConv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskConv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = F.relu(self.conv1(x))
        # out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MaskConv1d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = MaskConv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = MaskConv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskConv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, slice_size, num_classes, classes_per_task=[]):
#     #def __init__(self, block, num_blocks, num_classes):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         self.slice_size = slice_size

#         self.conv1 = MaskConv1d(2, 64, kernel_size=7, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.do = nn.Dropout(p=0.5)

#         if slice_size == 512:
#             self.avg_pool_out_dim = int(self.slice_size / 64)
#         else:
#             self.avg_pool_out_dim = int(self.slice_size / 16)
 
#         self.bn = nn.LayerNorm(512)

#         self.ds_module = Dempster_Shafer_module(
#                     n_feature_maps=512 * block.expansion,  # depends on your final feature dim
#                     n_classes=num_classes,
#                     n_prototypes=num_classes*20
#                 ) 

#         self.dm_layer = DM(num_class=num_classes, nu=0.9, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x, return_features=False):
#         # out = F.relu(self.conv1(x))
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool1d(out, self.avg_pool_out_dim)
#         out = out.view(out.size(0), -1)
#         features = self.do(out)

#         if torch.isnan(features).any():
#             raise ValueError("Model features contains NaNs")

#         features = self.bn(features)
 
#         if torch.isnan(features).any():
#             raise ValueError("Model norm features contains NaNs")
#         mass_normalized = self.ds_module(features)

#         if torch.isnan(mass_normalized).any():
#             raise ValueError("Model mass_normalized contains NaNs")

#         omegas = mass_normalized[:,-1]
#         beliefs = mass_normalized[:,:-1]

#         out = self.dm_layer(mass_normalized) # expected utility
#         if torch.isnan(out).any():
#             raise ValueError("Model dm_layer contains NaNs")
        
#         if return_features:
#             return out, features, omegas, beliefs
#         else:
#             return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, slice_size, num_classes, classes_per_task=None, nu=0.9):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.slice_size = slice_size
        self.classes_per_task = [6,6]

        self.conv1 = MaskConv1d(2, 64, kernel_size=7, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.do = nn.Dropout(p=0.3)

        if slice_size == 512:
            self.avg_pool_out_dim = int(self.slice_size / 64)
        else:
            self.avg_pool_out_dim = int(self.slice_size / 16)


        self.fc = nn.Linear(512 * block.expansion, num_classes) #nn.ModuleList(
        #     [nn.Linear(512 * block.expansion, task_classes) for task_classes in self.classes_per_task])

        # self.bn = nn.LayerNorm(512)
        # self.bn = nn.BatchNorm1d(512)


        # if self.classes_per_task is not None:
        #     # print("classes per task", classes_per_task)
        #     nus = [0.9,0.9,0.4]
        #     self.ds_module = nn.ModuleList([
        #         Dempster_Shafer_module(
        #             n_feature_maps=512 * block.expansion,
        #             n_classes=task_classes,
        #             n_prototypes=task_classes * 20
        #         )
        #         for task_classes in classes_per_task
        #     ])

        #     self.dm_layer = nn.ModuleList([
        #         DM(num_class=task_classes, nu=nus[i], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #         for i, task_classes in enumerate(classes_per_task)
        #     ])
        # else:
        #     self.ds_module = Dempster_Shafer_module(
        #             n_feature_maps=512 * block.expansion,
        #             n_classes=num_classes,
        #             n_prototypes=num_classes * 20
        #         )
        #     self.dm_layer = DM(num_class=num_classes, nu=0.9, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, task_id=None, return_features=False):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.do(out)
        out = self.layer2(out)
        out = self.do(out)
        out = self.layer3(out)
        out = self.do(out)
        out = self.layer4(out)
        out = self.do(out)
        out = F.avg_pool1d(out, self.avg_pool_out_dim)
        out = out.view(out.size(0), -1)
        out = self.do(out)
        # features = self.do(out)
        features = out

        if torch.isnan(features).any():
            raise ValueError("Model features contains NaNs")

        # features = self.bn(features)
        if return_features:
            return features
        if task_id is not None and isinstance(self.fc, nn.ModuleList):
            return self.fc[task_id](features)
        return self.fc(features)

        if torch.isnan(features).any():
            raise ValueError("Model norm features contains NaNs")
        # print(task_id)
        # print(x.shape)
        if task_id is not None:
            mass_normalized = self.ds_module[task_id](features)
            # print("mn", mass_normalized.shape)
            out = self.dm_layer[task_id](mass_normalized)
        else:
            mass_normalized = self.ds_module(features)
            out = self.dm_layer(mass_normalized)
        # print("out", out.shape)
        if torch.isnan(mass_normalized).any():
            raise ValueError("Model mass_normalized contains NaNs")

        omegas = mass_normalized[:, -1]
        beliefs = mass_normalized[:, :-1]

        if torch.isnan(out).any():
            raise ValueError("Model dm_layer contains NaNs")

        if return_features:
            return out, features, omegas, beliefs
        else:
            return out


def ResNet50_1d(slice_size, num_classes):
    print(1)
    return ResNet(Bottleneck, [3,4,6,3], slice_size, num_classes)


def ResNet18_1d(slice_size, num_classes, classes_per_task=None):
    return ResNet(BasicBlock, [2,2,2,2], slice_size, num_classes, classes_per_task=classes_per_task)

def ResNet34_1d(slice_size, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], slice_size, num_classes)

def ResNet101_1d(slice_size, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], slice_size,  num_classes)

def ResNet152_1d(num_classes, slice_size):
    return ResNet(Bottleneck, [3,8,36,3], slice_size, num_classes)

class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(MaskLinear, self).__init__(in_features, out_features, True)
        self.w_mask = Parameter(torch.ones(out_features, in_features))
        #self.b_mask = Parameter(torch.ones(out_features))

    def forward(self, input):
        return F.linear(input, self.weight*self.w_mask, self.bias)


class MaskConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.w_mask = Parameter(torch.ones(self.weight.shape))
        #self.b_mask = Parameter(torch.ones(self.bias.shape))
    
    def _conv_forward(self, input, weight):
        return F.conv1d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.w_mask)

class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
        #         padding, dilation, groups, bias, padding_mode)
        self.w_mask = Parameter(torch.ones(self.weight.shape))
        #self.b_mask = Parameter(torch.ones(self.bias.shape))
    
    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.w_mask)
