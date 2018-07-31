import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# UNet class

class Unet(nn.Module):
    def __init__(self, input_channels, nclasses):
        super(Unet, self).__init__()
        # go down
        self.conv1 = conv_bn_elu_dp(input_channels,32)
        self.conv2 = conv_bn_elu_dp(32, 64)
        self.conv3 = conv_bn_elu_dp(64, 128)
        self.conv4 = conv_bn_elu_dp(128, 256)
        self.conv5 = conv_bn_elu_dp(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # go up
        self.up_pool6 = up_pooling(512, 256)
        self.conv6 = conv_bn_elu_dp(512, 256)
        self.up_pool7 = up_pooling(256, 128)
        self.conv7 = conv_bn_elu_dp(256, 128)
        self.up_pool8 = up_pooling(128, 64)
        self.conv8 = conv_bn_elu_dp(128, 64)
        self.up_pool9 = up_pooling(64, 32)
        self.conv9 = conv_bn_elu_dp(64, 32)

        self.conv10 = nn.Conv2d(32, nclasses, 1)


        # test weight init
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()"""


    def forward(self, x):
        # normalize input data
        x = x/255.
        # go down
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # go up
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)


        output = self.conv10(x9)
        output = F.sigmoid(output)

        return output
