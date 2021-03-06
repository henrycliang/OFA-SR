from model import common
from model.modules.super_modules import SuperConv2d

import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return MultiEDSR_skip(args, dilated.dilated_conv)
    else:
        return MultiEDSR_skip(args)

class MultiEDSR_skip(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MultiEDSR_skip, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        self.n_feats=n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define the super tail module
        superconv=[SuperConv2d(in_channels=n_feats,out_channels=n_feats*16,
                               kernel_size=3,padding=1,bias=True)]

        # # define tail module
        # m_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     nn.Conv2d(
        #         n_feats, args.n_colors, kernel_size,
        #         padding=(kernel_size // 2)
        #     )
        # ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*superconv)
        self.out_conv=nn.Sequential(conv(n_feats,3,kernel_size))

        # The upsample layer directly from LR
        # self.upsampler=nn.Sequential(F.interpolate)

    def forward(self, x,scale,out_size):
        x1 = self.sub_mean(x)
        x = self.head(x1)

        res = self.body(x)
        res += x

        module=self.tail[0]

        # My define upsample layer
        x=module(res,scale**2*self.n_feats)
        x=nn.PixelShuffle(scale)(x)
        x=self.out_conv(x)

        # The upsample layer directly from LR
        x2=F.interpolate(x1,scale_factor=scale,mode='bicubic',align_corners=False)
        x=x+x2

        # downsample to the target scale
        x=F.interpolate(x,size=out_size,mode='bicubic',align_corners=False)


        x = self.add_mean(x)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

