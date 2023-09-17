import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os

def feature_vis(feats,output_shape,savedir,img_name,feature_name): # feaats形状: [b,c,h,w]
     channel_mean = torch.mean(feats,dim=1,keepdim=True) # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy() # 四维压缩为二维
     channel_mean = (((channel_mean - np.min(channel_mean))/(np.max(channel_mean)-np.min(channel_mean)))*255).astype(np.uint8)
     if not os.path.exists(savedir+'feature_vis/' + img_name ):
          os.makedirs(savedir+'feature_vis/' + img_name )
     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
     cv2.imwrite(savedir+'feature_vis/'+ img_name + '/' + feature_name + '.png',channel_mean)
