from model import common
from model.modules.super_modules import SuperConv2d
import torch

import torch.nn as nn

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return MultiEDSR_reshape(args, dilated.dilated_conv)
    else:
        return MultiEDSR_reshape(args)

class MultiEDSR_reshape(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MultiEDSR_reshape, self).__init__()

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
        superconv=[SuperConv2d(in_channels=n_feats,out_channels=100*16,
                               kernel_size=3,padding=1,bias=True)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*superconv)
        self.out_conv=nn.Sequential(conv(100,3,kernel_size))

    def forward(self, x,scale):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        module=self.tail[0]

        # My define upsample layer
        x=module(res,int(scale**2*100))
        x=self.pixel_shuffle(x,upscale_factor=scale)
        x=self.out_conv(x)

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

    def pixel_shuffle(self,input,scale,out_channels):
        shape=input.size()
        height=int(shape[0]*scale)
        width = int(shape[1] * scale)
        shuffle_map = torch.zeros((height, width, out_channels), dtype=torch.float32)
        for i in range(0, height):
            for j in range(0, width):
                for k in range(0,out_channels):
                    shuffle_map[i, j, k] = input[int(i//scale),int(j//scale),int(round(k*scale*scale+((i % scale) * scale + (j % scale) + 0.01)))]

        return shuffle_map



