from model import common
from model.modules.super_modules import SuperConv2d

import torch.nn as nn
import math
import torch
import torch.nn.functional as F

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return MultiEDSR_MetaShuffle(args, dilated.dilated_conv)
    else:
        return MultiEDSR_MetaShuffle(args)

class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,16)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

class MultiEDSR_MetaShuffle(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MultiEDSR_MetaShuffle, self).__init__()

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

        # position to vector
        self.P2W = Pos2Weight(inC=n_feats)

    def repeat_x(self,x,scale_int):
        N,C,H,W=x.size()
        x=x.view(N,C,H,1,W,1)

        x=torch.cat([x]*scale_int,3)
        x=torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)
        return x.contiguous().view(N,-1,C,H,W)

    # def repeat_weight(self, weight, scale, inw, inh):
    #     k = int(math.sqrt(weight.size(0)))
    #     outw = inw * scale
    #     outh = inh * scale
    #     weight = weight.view(k, k, -1)
    #     scale_w = (outw + k - 1) // k
    #     scale_h = (outh + k - 1) // k
    #     weight = torch.cat([weight] * scale_h, 0)
    #     weight = torch.cat([weight] * scale_w, 1)
    #
    #     weight = weight[0:outh, 0:outw, :]
    #
    #     return weight

    def forward(self, x,scale_int,out_size,pos_mat):
        # up-round to int_scale
        # scale_int=math.ceil(scale)
        x1 = self.sub_mean(x)
        x = self.head(x1)

        res = self.body(x)
        res += x

        module=self.tail[0]

        # My define upsample layer
        x=module(res,scale_int**2*self.n_feats)
        # partition the channels into 16 groups
        x_group=self.repeat_x(x,scale_int)
        x_group=x_group.view(x_group.size(0),x_group.size(1),self.n_feats,scale_int**2,x_group.size(-2),x_group.size(-1))

        # position vector
        local_vector=self.P2W(pos_mat.view(pos_mat.size(1),-1))
        local_vector=local_vector[:,:scale_int**2]
        local_weight=local_vector
        # local_weight = self.repeat_weight(local_vector, scale_int, x.size(2), x.size(3)) # repeat it to [outH,outW,16]
        # view local_weight to the target shape
        local_weight=local_weight.contiguous().view(x.size(2),scale_int,x.size(3),scale_int,-1,1) #将10000view成[50,2,50,2]时可能存在问题，要跟组成10000时的顺序一致。
        local_weight=local_weight.permute(1,3,0,2,4,5)
        local_weight=local_weight.contiguous().view(-1,x.size(2),x.size(3),local_weight.size(4),1)

        # view x_group to the target shape
        x_group=x_group.permute(0,1,4,5,2,3)

        out=torch.matmul(x_group,local_weight)
        out=out.contiguous().view(out.size(0),scale_int,scale_int,x.size(2),x.size(3),out.size(-2))
        out=out.permute(0,3,1,4,2,5)
        out=out.contiguous().view(out.size(0),scale_int*x.size(2),scale_int*x.size(3),out.size(-1))
        out=out.permute(0,3,1,2)

        out=self.out_conv(out)

        out = self.add_mean(out)

        return out

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

