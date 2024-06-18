import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from models.basic import *

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=1,ch_out=16)
        self.Conv2 = conv_block(ch_in=16,ch_out=32)
        self.Conv3 = conv_block(ch_in=32,ch_out=64)
        self.Conv4 = conv_block(ch_in=64,ch_out=128)
        self.Conv5 = conv_block(ch_in=128,ch_out=256)

    def forward(self,x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        return [x5, x4, x3, x2, x1]



class SharedDecoder(nn.Module):
    def __init__(self):
        super(SharedDecoder,self).__init__()
        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Up4 = up_conv(ch_in=128,ch_out=64)
        self.Up3 = up_conv(ch_in=64,ch_out=32)
        self.Up2 = up_conv(ch_in=32,ch_out=16)

        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.convg = single_conv(ch_in=16, ch_out=16)

    def forward(self,xs):
        # encoding path

        [x5, x4, x3, x2, x1] = xs

        # decoding + concat path
        d5 = F.interpolate(x5,(x4.size(2),x4.size(3)),mode='bilinear')
        d5 = self.Up5(d5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = F.interpolate(d5,(x3.size(2),x3.size(3)),mode='bilinear')
        d4 = self.Up4(d4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = F.interpolate(d4,(x2.size(2),x2.size(3)),mode='bilinear')
        d3 = self.Up3(d3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = F.interpolate(d3,(x1.size(2),x1.size(3)),mode='bilinear')
        d2 = self.Up2(d2)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        fs = self.convg(d2)

        return fs, [d2, d3, d4, d5]



class SpecificDecoder(nn.Module):
    def __init__(self):
        super(SpecificDecoder,self).__init__()

        self.Up5 = single_conv(ch_in=256, ch_out=128)
        self.Up4 = single_conv(ch_in=128, ch_out=64)
        self.Up3 = single_conv(ch_in=64, ch_out=32)
        self.Up2 = single_conv(ch_in=32, ch_out=16)

        self.Up_conv5 = single_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = single_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = single_conv(ch_in=32, ch_out=16)

    def forward(self, xs, ds):
        # encoding path

        [x5, x4, x3, x2, x1] = xs
        [d2, d3, d4, d5] = ds

        d_s5 = torch.cat((x4,d5),dim=1)
        d_s5 = self.Up5(d_s5)

        d_s4 = F.interpolate(d_s5,(d4.size(2), d4.size(3)),mode='bilinear')
        d_s4 = self.Up_conv5(d_s4)
        d_s4 = torch.cat((d_s4,d4),dim=1)
        d_s4 = self.Up4(d_s4)

        d_s3 = F.interpolate(d_s4,(d3.size(2), d3.size(3)),mode='bilinear')
        d_s3 = self.Up_conv4(d_s3)
        d_s3 = torch.cat((d_s3, d3),dim=1)
        d_s3 = self.Up3(d_s3)

        d_s2 = F.interpolate(d_s3, (d2.size(2), d2.size(3)),mode='bilinear')
        d_s2 = self.Up_conv3(d_s2)
        d_s2 = torch.cat((d_s2,d2),dim=1)
        fd = self.Up2(d_s2)

        return fd, [d_s2, d_s3, d_s4, d_s5]



class CU_Net(nn.Module):
    def __init__(self):
        super(CU_Net,self).__init__()
        
        self.shared_encoder = SharedEncoder()
        self.shared_decoder = SharedDecoder()

        self.blood_decoder = SpecificDecoder()
        self.choroid_decoder = SpecificDecoder()
        self.conv_blood = nn.Conv2d(32, 2,kernel_size=1,stride=1,padding=0)
        self.conv_choroid = nn.Conv2d(32, 3 ,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        
        xs = self.shared_encoder(x)
        fs, dss = self.shared_decoder(xs)

        fd_c, _ = self.choroid_decoder(xs, dss)
        fd_b, _ = self.blood_decoder(xs, dss)
    
        blood = self.conv_blood(torch.cat((fd_b,fs), dim=1))
        choroid = self.conv_choroid(torch.cat((fd_c,fs), dim=1))

        return torch.cat([choroid, blood], dim=1)

