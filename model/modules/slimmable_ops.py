import torch.nn as nn

class FLAGS(object):
    # width_mult_list=[0.25,0.5625,1.0]
    # width_mult_list = [0.09375, 0.125,   0.15625, 0.1875,  0.21875, 0.25,    0.28125, 0.3125,  0.34375,
    #                    0.375,   0.40625, 0.4375,  0.46875, 0.5,     0.53125, 0.5625,  0.59375, 0.625,
    #                    0.65625, 0.6875,  0.71875, 0.75,    0.78125, 0.8125,  0.84375, 0.875,   0.90625,
    #                    0.9375,  0.96875, 1.     ]
    # width_mult_list = [0.546875, 0.5625,   0.578125, 0.59375,  0.609375, 0.625,    0.640625, 0.65625,
    #                    0.671875, 0.6875,   0.703125, 0.71875,  0.734375, 0.75,     0.765625, 0.78125,
    #                    0.796875, 0.8125,   0.828125, 0.84375,  0.859375, 0.875,    0.890625, 0.90625,
    #                    0.921875, 0.9375,   0.953125, 0.96875,  0.984375, 1.      ]
    width_mult_list = [0.5, 0.75, 1.0]
    width_mult=1.0


class SlimmableConv2d(nn.Conv2d):
    def __init__(self,in_channels_list,out_channels_list,kernel_size,
                 stride=1,padding=0,dilation=1,groups_list=[1],bias=True):
        super(SlimmableConv2d,self).__init__(max(in_channels_list),max(out_channels_list),
                                             kernel_size,stride=stride,padding=padding,dilation=dilation,
                                             groups=max(groups_list),bias=bias)
        self.in_channels_list=in_channels_list
        self.out_channels_list=out_channels_list
        self.groups_list=groups_list
        if self.groups_list==[1]:
            self.groups_list=[1 for _ in range(len(in_channels_list))]
        self.width_mult=max(FLAGS.width_mult_list)

    def forward(self, input):
        self.width_mult=FLAGS.width_mult
        idx=FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels=self.in_channels_list[idx]
        self.out_channels=self.out_channels_list[idx]
        self.groups=self.groups_list[idx]
        weight=self.weight[:self.out_channels,:self.in_channels,:,:]
        if self.bias is not None:
            bias=self.bias[:self.out_channels]
        else:
            bias=self.bias
        y=nn.functional.conv2d(input,weight,bias,self.stride,self.padding,
                               self.dilation,self.groups)
        return y



## Use for Upsample Conv2d
class FLAGS_UpConv2d(object):
    # width_mult_list=[0.25,0.5625,1.0]
    # width_mult_list = [0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375,
    #                    0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.53125, 0.5625, 0.59375, 0.625,
    #                    0.65625, 0.6875, 0.71875, 0.75, 0.78125, 0.8125, 0.84375, 0.875, 0.90625,
    #                    0.9375, 0.96875, 1.]
    # width_mult_list = [0.546875, 0.5625, 0.578125, 0.59375, 0.609375, 0.625, 0.640625, 0.65625,
    #                    0.671875, 0.6875, 0.703125, 0.71875, 0.734375, 0.75, 0.765625, 0.78125,
    #                    0.796875, 0.8125, 0.828125, 0.84375, 0.859375, 0.875, 0.890625, 0.90625,
    #                    0.921875, 0.9375, 0.953125, 0.96875, 0.984375, 1.]
    width_mult_list = [0.5625, 1.0]
    width_mult=1.0

class SlimmableUpConv2d(nn.Conv2d):
    def __init__(self,in_channels_list,out_channels_list,kernel_size,
                 stride=1,padding=0,dilation=1,groups_list=[1],bias=True):
        super(SlimmableUpConv2d,self).__init__(max(in_channels_list),max(out_channels_list),
                                             kernel_size,stride=stride,padding=padding,dilation=dilation,
                                             groups=max(groups_list),bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult_out=max(FLAGS_UpConv2d.width_mult_list)
        self.width_mult_in =max(FLAGS.width_mult_list)

    def forward(self,input):
        self.width_mult_out=FLAGS_UpConv2d.width_mult
        self.width_mult_in =FLAGS.width_mult
        idx_in =FLAGS.width_mult_list.index(self.width_mult_in)
        idx_out=FLAGS_UpConv2d.width_mult_list.index(self.width_mult_out)
        self.in_channels=self.in_channels_list[idx_in]
        self.out_channels=self.out_channels_list[idx_out]
        self.groups = self.groups_list[idx_in]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding,
                                 self.dilation, self.groups)
        return y


## Use for Upsample Linear Layer
class FLAGS_UpLinear(object):
    width_mult_list=[0.5625,1.0]
    width_mult=1.0

class SlimmableUpLinear(nn.Linear):
    def __init__(self,in_features_list,out_features_list,bias=True):
        super(SlimmableUpLinear,self).__init__(max(in_features_list),
                                               max(out_features_list),
                                               bias=bias)
        self.in_features_list=in_features_list
        self.out_features_list=out_features_list
        self.width_mult=max(FLAGS_UpLinear.width_mult_list)

    def forward(self,input):
        self.width_mult=FLAGS_UpLinear.width_mult
        idx=FLAGS_UpLinear.width_mult_list.index(self.width_mult)
        self.in_features=self.in_features_list[idx]
        self.out_features=self.out_features_list[idx]
        weight= self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)
