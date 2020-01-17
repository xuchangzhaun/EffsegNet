# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import torch
from torch import nn
import torch.nn.functional as F
# import sys
# import torch.utils.model_zoo as model_zoo
# # package_path = '../input/efficientnetpython/'
# package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'
# sys.path.append(package_path)
from efficient_net.efficientnet_pytorch import EfficientNet
# backbone = EfficientNet.from_pretrained('efficientnet-b0')
# backbone.cuda()
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from efficient_net.efficientnet_pytorch.utils import round_filters, relu_fn
import collections

def _AsppConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn_momentum=0.1):
    asppconv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        nn.BatchNorm2d(out_channels, momentum=bn_momentum),
        nn.ReLU()
    )
    return asppconv

class AsppModule(nn.Module):
    def __init__(self, bn_momentum=0.1, output_stride=16):
        super(AsppModule, self).__init__()

        # output_stride choice
        if output_stride ==16:
            atrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            atrous_rates = 2*[0, 12, 24, 36]
        else:
            raise Warning("output_stride must be 8 or 16!")
        # atrous_spatial_pyramid_pooling part
        if output_stride == 8:
            input_channel = 48
        if output_stride == 16:
            input_channel = 136
        self._atrous_convolution1 = _AsppConv(input_channel, 256, 1, 1, bn_momentum=bn_momentum)
        self._atrous_convolution2 = _AsppConv(input_channel, 256, 3, 1, padding=atrous_rates[1], dilation=atrous_rates[1]
                                              , bn_momentum=bn_momentum)
        self._atrous_convolution3 = _AsppConv(input_channel, 256, 3, 1, padding=atrous_rates[2], dilation=atrous_rates[2]
                                              , bn_momentum=bn_momentum)
        self._atrous_convolution4 = _AsppConv(input_channel, 256, 3, 1, padding=atrous_rates[3], dilation=atrous_rates[3]
                                              , bn_momentum=bn_momentum)

        #image_pooling part
        self._image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(input_channel, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )

        self.__init_weight()

    def forward(self, input):
        input1 = self._atrous_convolution1(input)
        input2 = self._atrous_convolution2(input)
        input3 = self._atrous_convolution3(input)
        input4 = self._atrous_convolution4(input)
        input5 = self._image_pool(input)
        input5 = F.interpolate(input=input5, size=input4.size()[2:3][0], mode='bilinear', align_corners=True)
        return torch.cat((input1, input2, input3, input4, input5), dim=1)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Encoder(nn.Module):
    def __init__(self, bn_momentum=0.1, output_stride=16):
        super(Encoder, self).__init__()
        self.ASPP = AsppModule(bn_momentum=bn_momentum, output_stride=output_stride)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.dropout = nn.Dropout(0.5)

        self.__init_weight()

    def forward(self, input):
        input = self.ASPP(input)
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)
        input = self.dropout(input)
        return input


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(32, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()

        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256+48, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)

        self.conv5 = nn.Conv2d(40, 48, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(48, momentum=bn_momentum)
        self._init_weight()



    def forward(self, pool_16, low_level_feature,lower_2_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        lower_2_feature = self.conv5(lower_2_feature)
        lower_2_feature = self.bn5(lower_2_feature)

        x_4 = F.interpolate(pool_16, size=low_level_feature.size()[2:3][0], mode='bilinear' ,align_corners=True)
        x_5 = F.interpolate(lower_2_feature,size=low_level_feature.size()[2:3][0],mode='bilinear',align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class EffSegNet(nn.Module):
    def __init__(self, class_num,output_stride, pretrained, bn_momentum=0.1, freeze_bn=False):
        super(EffSegNet, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3',active = 'relu_fn')
        self.encoder = Encoder(bn_momentum,output_stride)
        self.decoder = Decoder(class_num,bn_momentum)
        self.output_stride = output_stride
        GlobalParams = collections.namedtuple('GlobalParams', [
            'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
            'num_classes', 'width_coefficient', 'depth_coefficient',
            'depth_divisor', 'min_depth', 'drop_connect_rate', ])
        #(1.2, 1.4, 300, 0.3)
        self.global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.3,
        drop_connect_rate=0.3,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=class_num,
        width_coefficient=1.2,
        depth_coefficient=1.4,
        depth_divisor=8,
        min_depth=None
        # image_size=image_size,
         )
        in_channels =10 # output of final block
        out_channels = round_filters(112, self.global_params)

        self._conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=self.global_params.batch_norm_momentum,eps=self.global_params.batch_norm_epsilon)

        # Final linear layers
        self._dropout = self.global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self.global_params.num_classes)


    def forward(self, input):
        #         lower_2_1_feature,lower_2_2_feature,lower_4_feature, pool_8_feature ,pool_16_feature = self.backbone(input)
        lower_2_1_feature,lower_2_2_feature,lower_4_feature,pool_8_feature,pool_16_feature = self.backbone(input)
        x = 0
        if self.output_stride == 8:
            x = self.encoder(pool_8_feature)
        if self.output_stride == 16:
            x = self.encoder(pool_16_feature)
        predict  = self.decoder(x,lower_4_feature,lower_2_1_feature)
        #分类
        predict = relu_fn(self._bn1(self._conv_head(predict)))
        # Pooling and final linear layer
        predict = F.adaptive_avg_pool2d(predict, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            predict = F.dropout(predict, p=self ._dropout, training=self.training)
        output = self._fc(predict)

        #分割
        # output = F.interpolate(predict,size = input.size()[2:3][0],mode='bilinear',align_corners=True)
        return output
    #     def freeze_bn(self):
    #         for m in self.modules():
    #             if isinstance(m, SynchronizedBatchNorm2d):
    #                 m.eval()
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.encoder, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p











