from model.position_encoding import positionalencoding2d,positionalencoding1d
import math
from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256,word_len=17,kernel_size=3):
        super().__init__()
        self.word_len = word_len
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim , in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim//2, in_dim)
        self.decode = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(word_len,1,stride=1,kernel_size=1,padding=0),
                                   )

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(B , C, H*W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        out = torch.matmul(word, x)
        out = out.reshape(B ,self.word_len, H, W)
        out = self.decode(out)

        # b, 1, 104, 104
        return out

class Fusion(nn.Module):
    def __init__(
            self,
            input_dim,
            norm_layer,
    ):
        super(Fusion, self).__init__()
        self.img_flo_fc = nn.Sequential(
            nn.Linear(input_dim * 2 , input_dim),
            nn.ReLU(inplace=True)
        )
        self.img_txt_fc = nn.Linear(input_dim + 512, input_dim)
        self.img_enhance_fc = nn.Linear(input_dim, input_dim)


    def forward(
            self,
            image,
            txt
    ):
        image_in = image
        txt = txt.mean(dim=1).unsqueeze(1)
        # flow = self.change_flow_dim(flow)
        img_avg = image.flatten(2).mean(dim=2)
        # [B, C]
        img_avg = img_avg.unsqueeze(1)
        # [B, 1, c]
        img_txt = torch.cat([img_avg, txt], dim=2)
        # [B, 1, c+512]
        img_txt_gate = torch.sigmoid(self.img_txt_fc(img_txt))

        img_txt_gate = img_txt_gate.squeeze(1).unsqueeze(2).unsqueeze(3)

        # image = image * img_txt_gate
        #0000000
        img_enhance = torch.sigmoid(self.img_enhance_fc(img_avg))

        img_enhance = img_enhance.squeeze(1).unsqueeze(2).unsqueeze(3)
        # # [B, c, 1, 1]
        image = image * img_enhance# we can do resudual in here
        image = image * img_txt_gate
        return image




class FPN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq


class FPN1(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256,word_len=17,kernel_size=3,in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super().__init__()
        self.word_len = word_len
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim//2, in_dim)
        self.decode = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(word_len,512,stride=1,kernel_size=1,padding=0),
                                   )

        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        self.norm_layer1 = nn.Sequential(nn.BatchNorm2d(out_channels[1]),
                                        nn.ReLU(True))
        self.norm_layer2 = nn.Sequential(nn.BatchNorm2d(out_channels[0]),
                                        nn.ReLU(True))
        self.norm_layer3 = nn.Sequential(nn.BatchNorm2d(out_channels[1]),
                                         nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[2],
                                 out_channels[2], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[0], out_channels[1], 3, 1)
        # aggregation
        self.aggr = nn.Sequential(conv_layer(3 * out_channels[1], out_channels[1], 1, 0),
                                  nn.BatchNorm2d(out_channels[1]),
                                  nn.ReLU())
        self.coordconv = nn.Sequential(
        CoordConv(3 * out_channels[1], 3 * out_channels[1], 3, 1),)
        self.coordconv3 = CoordConv(out_channels[1], out_channels[0], 3, 1)
        self.coordconv4 = CoordConv(out_channels[1], out_channels[1], 3, 1)
        self.coordconv5 = CoordConv(out_channels[2], out_channels[2], 3, 1)
        self.state1 = nn.Sequential(nn.Conv2d(out_channels[2],out_channels[1],kernel_size=1),
                                    nn.BatchNorm2d(out_channels[1]),
                                    nn.ReLU()
                                    )
        self.state2 = nn.Sequential(nn.Conv2d(out_channels[1], out_channels[0], kernel_size=1),
                                    nn.BatchNorm2d(out_channels[0]),
                                    nn.ReLU()
                                    )

    def forward(self, x, state):

        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = x
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        f5 = self.coordconv5(f5)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        # fusion 2: b, 512, 26, 26
        state1 = self.state1(state)
        # f4_ = self.f2_cat(torch.cat([f4,f5_],dim=1))
        f4 = self.f2_v_proj(v4)
        f4 = self.coordconv4(f4)
        f4 = self.norm_layer1(f4*state1)
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        state2 =self.state2(state1)
        f3 = self.coordconv3(v3)
        # f3 = self.f3_cat(torch.cat([f3, f4_], dim=1))
        f3 = self.norm_layer2(f3*state2)
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq3 = F.avg_pool2d(fq3, 2, 2)
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        # fq = self.norm_layer3(fq3*fq4*fq5)
        fq = self.coordconv(fq)
        fq = self.aggr(fq)

        # b, 512, 26, 26
        return fq
    #
    # def forward(self, imgs, state):
    #     # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
    #     v3, v4, v5 = imgs
    #     # fusion 1: b, 1024, 13, 13
    #     # text projection: b, 1024 -> b, 1024
    #     state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
    #     f5 = self.f1_v_proj(v5)
    #     f5 = self.norm_layer(f5 * state)
    #     # fusion 2: b, 512, 26, 26
    #     f4 = self.f2_v_proj(v4)
    #     f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
    #     f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
    #     # fusion 3: b, 256, 26, 26
    #     f3 = self.f3_v_proj(v3)
    #     f3 = F.avg_pool2d(f3, 2, 2)
    #     f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
    #     # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
    #     fq5 = self.f4_proj5(f5)
    #     fq4 = self.f4_proj4(f4)
    #     fq3 = self.f4_proj3(f3)
    #     # query
    #     fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
    #     fq = torch.cat([fq3, fq4, fq5], dim=1)
    #     fq = self.aggr(fq)
    #     fq = self.coordconv(fq)
    #     # b, 512, 26, 26
    #     return fq


class FPN2(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256,word_len=17,kernel_size=3,in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super().__init__()
        self.word_len = word_len
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim//2, in_dim)
        self.decode = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(word_len,512,stride=1,kernel_size=1,padding=0),
                                   )

        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        self.norm_layer1 = nn.Sequential(nn.BatchNorm2d(out_channels[1]),
                                        nn.ReLU(True))
        self.norm_layer2 = nn.Sequential(nn.BatchNorm2d(out_channels[0]),
                                        nn.ReLU(True))



        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[0], out_channels[1], 3, 1)
        # aggregation
        # self.aggr = conv_layer(4 * out_channels[1], out_channels[2], 1, 0)
        # layers_dim = out_channels[0] + out_channels[1] + out_channels[2]
        layers_dim =  out_channels[2]
        self.aggr = conv_layer(3 * out_channels[1] , out_channels[2], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(layers_dim, layers_dim, 3, 1),
                conv_layer(layers_dim, out_channels[2], 3, 1))
        self.state1 = nn.Sequential(nn.Conv2d(out_channels[2],out_channels[1],kernel_size=1),
                                    nn.BatchNorm2d(out_channels[1]),
                                    nn.ReLU()
                                    )
        self.state2 = nn.Sequential(nn.Conv2d(out_channels[1], out_channels[0], kernel_size=1),
                                    nn.BatchNorm2d(out_channels[0]),
                                    nn.ReLU()
                                    )
        # self.fuse_attn = Fuse_atten(out_channels[1])



    def forward(self, x,  state):

        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = x
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5)
        # f5 = self.fuse_attn_1(f5,state)
        # fusion 2: b, 512, 26, 26
        f4  = self.f2_v_proj(v4)
        # f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        state1 = self.state1(state)
        # f3 = self.fuse_attn_2(f4,state1)
        f4 = self.norm_layer1(f4)
        # f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        # fq4 = F.interpolate(f4,scale_factor=2,mode='bilinear')
        state2 =self.state2(state1)
        # f3  = self.fuse_attn_3(f3,state2)
        f3 = self.norm_layer2(f3)
        # f3 = self.f3_cat(torch.cat([f3, fq4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        # fq5,fq4,fq3 = self.fuse_attn(fq3,fq4,fq5)
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        # fq = fq3 + fq4 + fq5
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq
class FPN3(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256,word_len=17,kernel_size=3,in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super().__init__()
        self.word_len = word_len
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim//2, in_dim)
        self.decode = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(word_len,512,stride=1,kernel_size=1,padding=0),
                                   )

        # text projection
        self.txt_proj = linear_layer( out_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        self.norm_layer1 = nn.Sequential(nn.BatchNorm2d(out_channels[1]),
                                        nn.ReLU(True))
        self.norm_layer2 = nn.Sequential(nn.BatchNorm2d(out_channels[0]),
                                        nn.ReLU(True))



        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[0], out_channels[1], 3, 1)
        # aggregation
        # self.aggr = conv_layer(4 * out_channels[1], out_channels[2], 1, 0)
        # layers_dim = out_channels[0] + out_channels[1] + out_channels[2]
        layers_dim =  out_channels[2]
        self.aggr = conv_layer(3 * out_channels[1] , out_channels[2], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(layers_dim, layers_dim, 3, 1),
                conv_layer(layers_dim, out_channels[2], 3, 1))
        self.state1 = nn.Sequential(nn.Conv2d(out_channels[2],out_channels[1],kernel_size=1),
                                    nn.BatchNorm2d(out_channels[1]),
                                    nn.ReLU()
                                    )
        self.state2 = nn.Sequential(nn.Conv2d(out_channels[1], out_channels[0], kernel_size=1),
                                    nn.BatchNorm2d(out_channels[0]),
                                    nn.ReLU()
                                    )
        self.chen_enha = Image_enhance(out_channels[2])
        # self.fuse_attn = Fuse_atten(out_channels[1])



    def forward(self, x,  state):

        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = x
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1

        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        f5 = self.f1_v_proj(v5)
        # f5 = self.norm_layer(f5)
        # fusion 2: b, 512, 26, 26
        f4  = self.f2_v_proj(v4)
        # f4 = self.norm_layer1(f4)
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)

        # f3 = self.norm_layer2(f3)
        # f3 = self.f3_cat(torch.cat([f3, fq4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        # fq5,fq4,fq3 = self.fuse_attn(fq3,fq4,fq5)
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        # fq = fq3 + fq4 + fq5
        fq = self.aggr(fq)
        # fq = self.chen_enha(fq,state)
        fq = self.norm_layer(fq*state)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq
class Fuse_atten0(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.attention_fpn1= nn.MultiheadAttention(in_dim,num_heads=1,dropout=0.1)
        # self.attention_fpn2= nn.MultiheadAttention(in_dim,num_heads=8,dropout=0.1)
        # self.attention_fpn3= nn.MultiheadAttention(in_dim,num_heads=8,dropout=0.1)

        # self.fuse_attn_2= nn.MultiheadAttention(in_dim,num_heads=8,dropout=0.1)
        self.normlize1 = nn.LayerNorm(in_dim)

        self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.1)
        # self.dropout3 = nn.Dropout(0.1)

        # self.normlize3 = nn.LayerNorm(in_dim)
        # self.normlize4 = nn.BatchNorm2d(in_dim)
        # self.dropout2 = nn.Dropout(0.1)
        # self.fuse = Fusion()


    def forward(self,preds):
        b,c,h,w = preds.shape

        preds = preds.reshape(b,c,h*w).permute(2,0,1)
        preds_ori = preds
        # fq4 = fq3.reshape(b,c,h*w).permute(2,0,1)
        # fq5 = fq3.reshape(b,c,h*w).permute(2,0,1)
        # fq3_ori = fq3
        # fq4_ori = fq4
        # fq5_ori = fq5
        preds_attn,_ = self.attention_fpn1(preds,preds,preds)
        preds = preds * self.dropout1(preds_attn)
        preds = self.normlize1(preds + preds_ori)
        # fq5, _ = self.attention_fpn2(fq4, fq5, fq3)
        # fq4 = fq4_ori * self.dropout2(fq5)
        # fq3, _ = self.attention_fpn3(fq3, fq4, fq5)
        # fq3 = fq3_ori * self.dropout3(fq5)
        # fq5 = self.normlize1(fq5+fq5_ori)
        # fq4 = self.normlize1(fq4+fq4_ori)
        # fq3 = self.normlize1(fq3+fq3_ori)
        #
        # fq5 = fq5.permute(1,2,0).reshape(b,c,h,w)
        # fq4 = fq4.permute(1,2,0).reshape(b,c,h,w)
        # fq3 = fq3.permute(1,2,0).reshape(b,c,h,w)

        # state = state.reshape(b,c,1).permute(2,0,1)
        # state_attn,_ = self.fuse_attn_1(state,img,img)
        # img_attn,_ = self.fuse_attn_2(img,state,state)
        # state_atten = state + self.dropout1(state_attn)
        # state_atten = self.normlize3(state_atten)
        # img_atten = img + self.dropout2(img_attn)
        # img_atten = self.normlize3(img_atten)
        # img_final = (state_atten * img_atten).
        # img_ori = self.normlize4(img_final)
        return preds


class Fuse_atten(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.direct = nn.Sequential(nn.Linear(1024,in_dim ),
                                    nn.BatchNorm1d(in_dim ),
                                    nn.Sigmoid())


    def forward(self,preds,state):
        b,c,h,w = preds.shape
        preds = preds.reshape(b,c,-1)
        state_dir = self.direct(state)
        state_dir = state_dir.reshape(b,1,self.in_dim)
        preds = torch.matmul(state_dir,preds)
        preds = preds.reshape(b,1,h,w)

        return preds

class Image_enhance(nn.Module):
    def __init__(
            self,
            input_dim,
    ):
        super(Image_enhance, self).__init__()

        self.img_enhance_fc = nn.Linear(input_dim, input_dim)
        self.img_txt_fc = nn.Linear(input_dim + 1024, input_dim)



    def forward(
            self,
            image,
            state
    ):
        # flow = self.change_flow_dim(flow)
        img_avg = image.flatten(2).mean(dim=2)
        # [B, C]
        state  = state.squeeze(2).permute(0,2,1)
        img_avg = img_avg.unsqueeze(1)
        # [B, 1, c]
        img_txt = torch.cat([img_avg, state], dim=2)
        # [B, 1, c+512]
        img_txt_gate = torch.sigmoid(self.img_txt_fc(img_txt))
        img_txt_gate = img_txt_gate.squeeze(1).unsqueeze(2).unsqueeze(3)
        img_enhance = torch.sigmoid(self.img_enhance_fc(img_avg))

        img_enhance = img_enhance.squeeze(1).unsqueeze(2).unsqueeze(3)
        # # [B, c, 1, 1]
        image = image * img_enhance# we can do resudual in here
        image = image * img_txt_gate
        return image

class U_B_Fuse(nn.Module):
    def __init__(
            self,
            input_dim,
    ):
        super(U_B_Fuse, self).__init__()

        self.img_enhance_fc = nn.Linear(input_dim, input_dim)
        self.img_txt_fc = nn.Linear(input_dim + 1024, input_dim)
        self.txt_enhance_fc = nn.Linear(input_dim, input_dim)
        self.txt_img_fc = nn.Linear(input_dim + 1024, input_dim)



    def forward(
            self,
            image,
            state
    ):
        # flow = self.change_flow_dim(flow)
        img_avg = image.flatten(2).mean(dim=2)
        # [B, C]
        img_avg = img_avg.unsqueeze(1)
        state = state.unsqueeze(1)
        stata_ori = state
        # [B, 1, c]
        img_txt = torch.cat([img_avg, state], dim=2)
        # [B, 1, c+512]
        img_txt_gate = torch.sigmoid(self.img_txt_fc(img_txt))
        img_txt_gate = img_txt_gate.squeeze(1).unsqueeze(2).unsqueeze(3)
        img_enhance = torch.sigmoid(self.img_enhance_fc(img_avg))
        img_enhance = img_enhance.squeeze(1).unsqueeze(2).unsqueeze(3)
        # state
        txt_img_gate = torch.sigmoid(self.txt_img_fc(img_txt))
        # txt_img_gate = txt_img_gate.squeeze(1).unsqueeze(2).unsqueeze(3)
        txt_enhance = torch.sigmoid(self.txt_enhance_fc(state))
        # txt_enhance = txt_enhance.squeeze(1).unsqueeze(2).unsqueeze(3)
        # # [B, c, 1, 1]
        image = image * img_enhance# we can do resudual in here
        image = image * img_txt_gate
        # state = state * txt_enhance
        # state = state * txt_enhance
        # state = state * txt_img_gate
        state = state.squeeze(1).unsqueeze(2).unsqueeze(3)
        image = image * state
        state = state.squeeze(3).squeeze(2)
        return image, state