import torch
import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns

__all__ = ['AsymFusionModule']


class AsymFusionModule(nn.Module):
    def __init__(self, planes_high, planes_low, planes_out):  #
        super(AsymFusionModule, self).__init__()
        self.pa = nn.Sequential(    #Pixel Attention
            nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.BatchNorm2d(planes_low//4),
            nn.ReLU(True),
            ##
            antialiased_cnns.BlurPool(planes_low//4,stride=1),

            nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.plus_conv = nn.Sequential(
            nn.Conv2d(planes_high, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True),
            antialiased_cnns.BlurPool(planes_low,stride=1)
        )
        self.ca = nn.Sequential(    #channel attention
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.BatchNorm2d(planes_low//4),
            nn.ReLU(True),
            antialiased_cnns.BlurPool(planes_low//4,stride=1),

            nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.end_conv = nn.Sequential(
            nn.Conv2d(planes_low, planes_out, 3, 1, 1),
            nn.BatchNorm2d(planes_out),
            nn.ReLU(True),
            antialiased_cnns.BlurPool(planes_out,stride=1),
        )

    def forward(self, x_high, x_low):
        x_high = self.plus_conv(x_high)
        pa = self.pa(x_low)
        ca = self.ca(x_high)

        feat = x_low + x_high
        feat = self.end_conv(feat)
        feat = feat * ca
        feat = feat * pa
        return feat