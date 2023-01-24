import numpy as np

import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision.models as models

model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20, pretrained=True):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        self.features = models.alexnet(pretrained).features
        for name in self.features.modules():
            name[12] = nn.Identity()
            break
        self.classifier = nn.Sequential(
                            nn.Conv2d(256, 256, (3, 3), (1, 1)),
                            nn.ReLU(True),
                            nn.Conv2d(256, 256, (1, 1), (1, 1)),
                            nn.ReLU(True),
                            nn.Conv2d(256, num_classes, (1, 1), (1, 1)))
        #self.max_pool = nn.MaxPool2d((13, 13), (1, 1), (0, 0))

    def forward(self, x):
        #TODO: Define forward pass
        heatmap = self.classifier(self.features(x))
        imoutput = nn.functional.max_pool2d(heatmap, heatmap.shape[2:4], (1,1))
        imoutput = imoutput.reshape((imoutput.shape[0], -1))
        return heatmap, imoutput


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20, pretrained=True):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Define model
        self.features = models.alexnet(pretrained).features
        for name in self.features.modules():
            name[12] = nn.Identity()
            break
        self.classifier = nn.Sequential(
                            nn.Conv2d(256, 256, (3, 3), (1, 1)),
                            nn.ReLU(True),
                            nn.Conv2d(256, 256, (1, 1), (1, 1)),
                            nn.ReLU(True),
                            nn.Conv2d(256, num_classes, (1, 1), (1, 1)))


    def forward(self, x):
        #TODO: Define fwd pass
        heatmap = self.classifier(self.features(x))

        x0 = nn.functional.max_pool2d(heatmap, heatmap.shape[2:4], (1, 1))
        x1 = nn.functional.avg_pool2d(heatmap, heatmap.shape[2:4], (1, 1))
        x2 = nn.functional.max_pool2d(heatmap, (heatmap.shape[2]//2, heatmap.shape[3]//2), (2, 2), padding=(1,1))
        x3 = nn.functional.avg_pool2d(x2, x2.shape[2:3], (1,1))

        imoutput = x0 + x1 # + x3
        imoutput = imoutput.reshape((imoutput.shape[0], -1))
        return heatmap, imoutput


def initialize_weights_xavier(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight.data)

def localizer_alexnet(pretrained=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(pretrained=pretrained, **kwargs)
    #TODO: Initialize weights correctly based on whether it is pretrained or not
    if pretrained == False:
        model.features.apply(initialize_weights_xavier)
    model.classifier.apply(initialize_weights_xavier)

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(pretrained=pretrained, **kwargs)
    #TODO: Ignore for now until ins
    if pretrained == False:
        model.features.apply(initialize_weights_xavier)
    model.classifier.apply(initialize_weights_xavier)    

    return model