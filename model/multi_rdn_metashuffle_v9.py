
# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
import time
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from model.modules.super_modules import SuperConv2d,SuperLinear

def make_model(args, parent=False):
    return MultiRDN_MetaShuffle(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c* G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(inplace=True)
        )
        self.SuperLinear=nn.Sequential(SuperLinear(256,16))
        self.softmax = nn.Softmax()

    def forward(self,x,num_channels):

        x = self.meta_block(x)
        LinearModule = self.SuperLinear[0]
        output=LinearModule(x,num_channels)
        output = self.softmax(output)
        return output


class MultiRDN_MetaShuffle(nn.Module):
    def __init__(self, args,conv=common.default_conv):
        super(MultiRDN_MetaShuffle, self).__init__()
        r = args.scale[0]
        self.G0 = args.G0
        G0=args.G0
        kSize = args.RDNkSize
        self.scale = 1
        self.args = args
        self.scale_idx = 0
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        superconv = [SuperConv2d(in_channels=G0, out_channels=G0 * 16,
                                 kernel_size=3, padding=1, bias=True)]
        self.tail = nn.Sequential(*superconv)
        self.out_conv = nn.Sequential(conv(G0, 3, kernel_size=3))

        # position to vector
        self.P2W = Pos2Weight(inC=G0)

    def repeat_x(self,x,scale_int):
        N,C,H,W=x.size()
        x=x.view(N,C,H,1,W,1)

        x=torch.cat([x]*scale_int,3)
        x=torch.cat([x]*scale_int,5)
        return x.contiguous().view(N,C,H*scale_int,W*scale_int)

    
    def forward(self, x, scale,out_size,pos_mat_dict):
        #d1 =time.time()
        pos_mat = pos_mat_dict['pos_mat']
        mask = pos_mat_dict['mask'].squeeze(0)
        scale_int=math.ceil(scale)
        num_channels=math.ceil(scale_int**2)
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        ###############################################################################################################
        # 至此前面的这一部分是LR feature map的部分

        module = self.tail[0]
        x = module(x, num_channels * self.G0)
        x_group = F.interpolate(x,size=out_size,mode='bilinear',align_corners=True)
        # x_group = self.repeat_x(x, scale_int)
        # x_group=torch.masked_select(x_group,mask.to('cuda'))
        # x_group=x_group.contiguous().view(x.size(0),self.G0*num_channels,out_size[0],out_size[1])
        x_group=x_group.permute(0,2,3,1)
        x_group=x_group.contiguous().view(x_group.size(0),x_group.size(1),x_group.size(2),self.G0,num_channels)

        # position vector
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1), num_channels)
        # view local_weight to the target shape
        local_weight = local_weight.contiguous().view(x_group.size(1), x_group.size(2), -1, 1)


        x_group = torch.matmul(x_group, local_weight)
        x_group = x_group.squeeze(-1)
        x_group = x_group.permute(0, 3, 1, 2)


        x_group = self.out_conv(x_group)
        x_group = self.add_mean(x_group)

        return x_group



