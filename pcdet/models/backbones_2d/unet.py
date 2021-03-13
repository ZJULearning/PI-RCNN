import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils.seg_loss import FocalLoss


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, bn=True):
        super(DoubleConv, self).__init__()
        if bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self, in_ch, out_ch, bn=True):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, bn=bn),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, bn=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch, bn=bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        self.input_channel = self.model_cfg.get('INPUT_CHANNELS', None)
        bn = self.model_cfg.get('BN', None)
        channels = self.model_cfg.get('CHANNELS', None)
        self.output_channels = self.model_cfg.get('OUTPUT_CHANNELS', None)

        self.inc = DoubleConv(self.input_channel, channels[0], bn=bn)
        self.down1 = Down(channels[0], channels[1], bn=bn)
        self.down2 = Down(channels[1], channels[2], bn=bn)
        self.down3 = Down(channels[2], channels[3], bn=bn)
        self.down4 = Down(channels[3], channels[3], bn=bn)
        self.up1 = Up(channels[4], channels[2], bn=bn)
        self.up2 = Up(channels[3], channels[1], bn=bn)
        self.up3 = Up(channels[2], channels[0], bn=bn)
        self.up4 = Up(channels[1], channels[0], bn=bn)

        self.out_conv = nn.Conv2d(channels[0], self.output_channels, 1)

        output_prob = self.model_cfg.get('OUTPUT_PROB', None)
        self.output_layer = None
        if output_prob:
            self.output_layer = nn.Softmax(dim=1)

        self.loss_func = None
        if hasattr(self.model_cfg, 'LOSS_CONFIG'):
            apply_nonlin = nn.Softmax(dim=1) if not output_prob else None
            if self.model_cfg.LOSS_CONFIG.NAME == 'FocalLoss':
                self.loss_func = FocalLoss(apply_nonlin=apply_nonlin, alpha=getattr(self.model_cfg.LOSS_CONFIG, 'ALPHA', None))
            else:
                raise NotImplementedError

        self.forward_data_dict = None

    def get_output_feature_dim(self):
        return self.output_channels

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        pred = self.forward_data_dict['pred_image_seg']
        target = self.forward_data_dict['image_seg_label']
        loss = self.loss_func(pred, target)
        tb_dict.update({'image_seg_loss': loss})
        return loss, tb_dict

    def forward(self, data_dict):
        x = data_dict['image']
        x = x.permute(0, 3, 1, 2)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.out_conv(x)

        if self.output_layer is not None:
            out = self.output_layer(out)

        data_dict['pred_image_seg'] = out

        self.forward_data_dict = data_dict

        return data_dict
