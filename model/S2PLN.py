import torch
import thop
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, conv3x3
import sys
import numpy as np
from torch.autograd import Variable
import random
import os
from skimage.feature import local_binary_pattern

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # change your path
    model_path = '/home/ywj/paper/S2PLN/pretrained_model/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    # print(model)
    return model

class Feature_Generator_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Generator_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

    def forward(self, input):
        outs = []
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)   # (n, 64, 112, 112)
        outs.append(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)   # (n, 64, 56, 56)
        outs.append(feature)
        feature = self.layer2(feature)   # (n, 128, 28, 28)
        outs.append(feature)
        feature = self.layer3(feature)   # (n, 256, 14, 14)
        outs.append(feature)
        return feature, outs

class Feature_Embedder_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Embedder_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, outs, norm_flag=True):
        feature = self.layer4(input)       # (n,512,7,7)
        outs.append(feature)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)   # (n,512)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature, outs


class Classifier(nn.Module):
    def __init__(self, cls_num):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, cls_num)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DII_Generator(nn.Module):
    def __init__(self, model):
        super(DII_Generator, self).__init__()
        if(model == 'resnet18'):
            self.backbone = Feature_Generator_ResNet18()
            self.embedder = Feature_Embedder_ResNet18()
        else:
            print('Wrong Name!')

    def forward(self, input, norm_flag=True):
        feature, outs = self.backbone(input)
        feature, outs = self.embedder(feature, outs, norm_flag)
        return feature, outs

class S_model(nn.Module):
    def __init__(self, model):
        super(S_model, self).__init__()
        if(model == 'resnet18'):
            self.backbone = Feature_Generator_ResNet18()
            self.embedder = Feature_Embedder_ResNet18()
        else:
            print('Wrong Name!')
        self.classifier = Classifier(2)

    def forward(self, input, norm_flag=True):
        feature, outs = self.backbone(input)
        feature, outs = self.embedder(feature, outs, norm_flag)
        classifier_out = self.classifier(feature, norm_flag)
        return classifier_out, feature, outs


if __name__ == '__main__':
    # x = Variable(torch.ones(6, 3, 224, 224))
    # model = S_model('resnet18')
    # y, v, outs = model(x, True)
    # print(len(outs))

    model = Classifier(2)
    print(model)


    # flops, params = thop.profile(model, inputs=(x, True))
    # flops, params = thop.clever_format([flops, params], "%.3f")
    # print("flops = ", flops)
    # print("params = ", params)






