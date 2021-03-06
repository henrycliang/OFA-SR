import torch
from torch import nn
from torch.nn import functional as F

class SuperConv2d(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,
                 padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
        super(SuperConv2d,self).__init__(in_channels,out_channels,kernel_size,
                                         stride,padding,dilation,groups,bias,padding_mode)

    def forward(self,x,config):
        in_nc=x.size(1)
        out_nc=config
        weight=self.weight[:out_nc,:in_nc]
        if self.bias is not None:
            bias=self.bias[:out_nc]
        else:
            bias=None
        return F.conv2d(x,weight,bias,self.stride,self.padding,self.dilation,self.groups)


class SuperLinear(nn.Linear):
    def __init__(self,in_features,out_features,bias=True):
        super(SuperLinear,self).__init__(in_features,out_features,bias)

    def forward(self,input,num_channels):
        weight=self.weight[:num_channels,:]
        if self.bias is not None:
            bias=self.bias[:num_channels]
        else:
            bias=None
        return F.linear(input,weight,bias)




