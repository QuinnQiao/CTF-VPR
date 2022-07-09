import torchvision
import torch
from torch import nn


__all__ = ['vgg']


class VGG(nn.Module):
    __factory = {
        16: torchvision.models.vgg16,
    }

    __layer_index = { # vgg16
        'conv5':24,
        'conv4':17,
        'conv3':10,
        'conv2':5,
        'conv1':0
    }

    def __init__(self, depth, pretrained=None):
        super(VGG, self).__init__()
        self.depth = depth
        self.feature_dim = 512
        if depth not in VGG.__factory:
            raise KeyError("Unsupported depth:", depth)
        vgg = VGG.__factory[depth](pretrained=False)
        layers = list(vgg.features.children())[:-2]
        self.base = nn.Sequential(*layers) # capture only feature part and remove last relu and maxpool
        if pretrained is not None:
            self.base.load_state_dict(torch.load(pretrained))
            print('Load from {}'.format(pretrained))

    def forward(self, x):
        x = self.base(x)
        return x


def vgg(pretrained=None):
    return VGG(16, pretrained)
