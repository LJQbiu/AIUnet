import logging

import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from model.u2net import u2net_full

from .layers import FPN, Projector, Fusion,FPN1,FPN2,Fuse_atten,FPN3
from utils.vis_feature import feature_vis

class AIUnet(nn.Module):
    def __init__(self, cfg):
        super(AIUnet, self).__init__()
        # vis_dim = [64,64,128,256,512,512]
        # out_vis_dim = [64,64,128,256,512,512]
        out_vis_dim = [64,64,128,256,512,1024]
        out_vis_dim_len = len(out_vis_dim)
        self.size = cfg.input_size
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain1,
                                    map_location="cpu").eval()
        self.backbone1 = build_model(clip_model.state_dict(), cfg.word_len).float()
        clip_model = torch.jit.load(cfg.clip_pretrain2,
                                    map_location="cpu").eval()
        self.backbone2 = build_model(clip_model.state_dict(), cfg.word_len).float()
        for param in self.backbone1.parameters():
            param.requires_grad = True
        self.backbone1.train()
        for param in self.backbone2.parameters():
            param.requires_grad = True
        self.backbone2.train()
        del clip_model
        loguru.logger.info("freezed clip parameters")
        # Multi-Modal FPN
        self.neck = FPN3(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # self.neck = FPN(cfg.word_dim, 1024,cfg.word_len, 3,in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # u2net encoder
        self.u2net_encoder = u2net_full(out_ch=1)
        # Projector
        self.proj = nn.ModuleList()
        for i in range(out_vis_dim_len):
            # self.proj.append(Projector (cfg.word_dim, cfg.vis_dim // 2, 3))
            self.proj.append(Projector(cfg.word_dim, out_vis_dim[i],cfg.word_len, 3))
        # self.proj_0 = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        pre_indim = out_vis_dim_len-1
        self.preduce_pre = nn.Sequential(
                                         nn.Conv2d(pre_indim,1,stride=1,padding=0,kernel_size=1),
        )
        # self.initial_parameters()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.heatmap = cfg.feature_vis
        self.heatmap_savedir = cfg.heatmap_save
        self.pre_weight = nn.Sequential(
            nn.Linear(6,6),
            nn.Sigmoid()
        )
        self.defuse = Fuse_atten(in_dim=pre_indim)



    def forward(self, img, word, mask=None,img_name=None):
        '''
            img: b, 3, h, w
            flow: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w

        '''

        # vis: C3 / C4 / C5 ||  C3: b,512,H/8,W/8  ||  C4: b,512,H/16,W/16  ||  C5: b,512,H/32,W/32
        # word: b, length, 512
        # state: b, 512
        vis = self.backbone1.encode_image(img)
        word, state = self.backbone2.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq_o = self.neck(vis, state)
        # fq_o = self.neck(vis, word, state)
        # 0: b, 64, 26, 26
        # 1: b, 64, 26, 26
        # 2: b, 128, 26, 26
        # 3: b, 256, 26, 26
        # 4: b, 512, 26, 26
        # 5: b, 512, 26, 26
        fq_net,state = self.u2net_encoder(fq_o,word,state)
        preds = []
        i = 0
        # resize mask
        # preweight = self.pre_weight(torch.tensor([1,1,1,1,1,1],dtype=torch.float).cuda())
        for i,fq_i in enumerate(fq_net):
            pred_i = self.proj[i](fq_i,word)

            if i >= len(fq_net) - 1:
                pass
            else:
                # preds.append(pred_i)
                preds.append(F.interpolate(pred_i, [self.size, self.size], mode='bilinear'))

        pred = torch.cat(preds, 1)

        if self.training:
            loss_bec = 0
            loss_dice = 0
            pre_hight = len(preds)
            # for i,pred_assist in enumerate(preds):
            #     if i <= 2:
            #         loss_bec = loss_bec + F.binary_cross_entropy_with_logits(pred_assist,mask)
            #         loss_dice = loss_dice + dice_loss(pred_assist,mask)

            # we can add time feature in here, so that we can devide the score of every layer
            # pred = self.preduce_pre(pred)
            pred = self.defuse(pred, state)
            # vis_sen = fq_net[-1]
            # vis_sen = vis_sen.squeeze()
            # pdist = nn.PairwiseDistance(p=2)
            # p_dist = pdist(vis_sen, state)
            # loss_aline = torch.mean(p_dist)

            loss_pre_bec = F.binary_cross_entropy_with_logits(pred,mask)
            loss_pre_dice = dice_loss(pred,mask)
            loss_pre = loss_pre_bec + loss_pre_dice
            loss = loss_pre + loss_bec + loss_dice


            return pred, mask, loss
        else:

            # pred_mask = F.interpolate(pred, [320, 320]).squeeze().detach()



            # fq net
            if self.heatmap == True:
                feature_vis(fq_o, [self.size, self.size], self.heatmap_savedir, img_name, 'fq_o')
                for id, feature in enumerate(vis):
                    feature_name = f'vis{id}'
                    feature_vis(feature, [self.size, self.size], self.heatmap_savedir, img_name, feature_name)
                for id,feature_fq_net in enumerate(fq_net):
                    feature_name = f'fq_net{id}'
                    feature_vis(feature_fq_net, [self.size, self.size], self.heatmap_savedir, img_name, feature_name)

                #preds befor cat
                for id,feature_fq_net in enumerate(preds):
                    feature_name = f'preds{id}'
                    feature_vis(feature_fq_net, [self.size, self.size], self.heatmap_savedir, img_name, feature_name)

                # pred before decode

                feature_vis(pred, [self.size, self.size], self.heatmap_savedir, img_name, 'pred')

            # pred = self.preduce_pre(pred)
            pred = self.defuse(pred, state)

            return pred.squeeze()


    def initial_parameters(self):
        initial = nn.init.normal_
        mean = 0
        std = 1
        for m in self.neck.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                initial(m.weight, std=std)
        for m in self.txt.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                initial(m.weight, std=std)
        for m in self.u2net_encoder.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                initial(m.weight, std=std)
        for m in self.proj.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                initial(m.weight, std=std)
        for m in self.preduce_pre.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                initial(m.weight, std=std)


def _sigmoid(x):
    y = torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)
    return y

def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = _sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = _sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()