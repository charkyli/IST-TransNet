from numpy import append
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18,resnet34
from .context import CPM, AGCB_Element, AGCB_Patch
from .fusion import *
from .cbam import CBAMBlock,CBAM
from .transformer import  *
from .vit import get_vit
import antialiased_cnns
__all__ = ['transnet']


class _FCNHead(nn.Module):
    
    # def __init__(self, in_channels, out_channels, drop=0.5):
    #     super(_FCNHead, self).__init__()
    #     inter_channels = in_channels // 4
    #     self.block = nn.Sequential(
    #         nn.Conv2d(in_channels, inter_channels, 3, 1, 1),
    #         nn.BatchNorm2d(inter_channels),
    #         nn.ReLU(True),
    #         nn.Dropout(drop),
    #         nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
    #     )
    def __init__(self,in_channels,out_channels,kernel_size=3,upsampling=1):
        super(_FCNHead,self,).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=kernel_size//2),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling >1 else nn.Identity()
        )
    def forward(self, x):
        return self.block(x)


class ISTTransNet(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',drop=0.5
                ):
        super(ISTTransNet, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=False)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(1280, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1)
        self.vit_branch = get_vit(num_classes=2)
        self.conv1 = nn.Conv2d(512,512,kernel_size=1,stride=2)
        self.conv2 = nn.Conv2d(512,1024,kernel_size=3,stride=1)
        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)
        #添加transformer，
        #self.trans = Transformer(num_encoder_layers=4,d_model=512)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _,_, hei, wid = x.shape
        #mid_feature = []
        c1, c2, c3 = self.backbone(x)
        vit_in = self.conv2(self.conv1(c3))
        vit_out = self.vit_branch(vit_in) 
        
        cpm_out = self.context(c3)
        
        #mid_feature.append(cpm_out)  #context为CPM模块
        vit_out = torch.cat((cpm_out,vit_out),dim=1)  #将vit输出的结果和cpm结果进行拼接

        #no cpm
        #out = torch.cat((vit_out,c3),dim=1)
        out = F.interpolate(vit_out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        #mid_feature.append(out)

        out = self.fuse23(out, c2)
        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        
        #mid_feature.append(out)
        out = self.fuse12(out, c1) #--->[8,128,128,128]
 
        out = self.head(out)
        out = F.interpolate(out, size=[hei, wid], mode='bilinear', align_corners=True)
        #mid_feature.append(out)
        
        return out




class ISTTrans_Pro(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1):
        super(ISTTrans_Pro, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape

        c1, c2, c3 = self.backbone(x)

        out = self.context(c3)

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out


def transnet(backbone, scales, reduce_ratios, gca_type, gca_att, drop):
    return ISTTransNet(backbone=backbone, scales=scales, reduce_ratios=reduce_ratios, gca_type=gca_type, gca_att=gca_att, drop=drop)
