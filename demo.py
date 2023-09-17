from model.segment_anything import sam_model_registry
from model.segmenter import  RSAM, AIUNET_ORI,DINOSAM
from model.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from model.aiunet import AIUNET
from model import build_segmenter
import argparse
from utils.dataset import tokenize
import utils.config as config
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import  cv2
from utils.vis_mask import  vis
import numpy as np
import torch.nn.functional as F
from loguru import logger
def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='config/refcoco/cris_r101.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def demo(img_name,promote,model_name):
    cfg = get_parser()
    root_path = './demo'
    img_url =  os.path.join(root_path,img_name)
    img_name, _ = img_name.split('.')
    model = load_models(cfg,model_name)


    logger.info("Start to process img...")
    img_t,img_ori,ori_size = loadimg(img_url,cfg)
    logger.info("Start to process text...")
    txt,promote = loadtxt(promote,cfg.word_len)
    logger.info("Start to predict mask...")
    pred = genrate_mask(img_t,txt,img_ori,cfg,rsam=model)
    logger.info("Start visualizer mask...")
    save_mask(root_path,img_name,img_ori,pred,ori_size)

    root_path = os.path.join(root_path, img_name)
    # for i,pred in enumerate(pred2):
    #     img_name = f'{i}'
    #     # save_mask(root_path,img_name,img_ori,pred['segmentation'],ori_size)
    #     save_mask(root_path,img_name,img_ori,pred['masks'],ori_size)
    print("Done!")

def load_models(cfg,name=None):
    if name == 'RSAM':
        logger.info("Loading model RSAM...")
        model, _ = build_segmenter(DINOSAM, cfg)
        state_dict = torch.load(cfg.resume)['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'module' in key:
                key = key.replace('module.', '')
            new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        model.to(device=cfg.device)
        model.eval()
        for name,param in model.named_parameters():
            param.requires_grad = False
        model.eval()
        del new_state_dict
        return model

    elif name == "AIUNET_ORI":
        logger.info("Loading model aiunet...")
        aiunet,_ = build_segmenter(AIUNET_ORI,cfg)
        state_dict = torch.load(cfg.AIunet_path)['state_dict']
        new_state_dict = { }
        for key, value in state_dict.items():
            if 'module' in key:
                key = key.replace('module.','')
            new_state_dict[key] = value
        aiunet.load_state_dict(new_state_dict)
        aiunet.to(device=cfg.device)
        aiunet.eval()
        del new_state_dict
        return aiunet
    else:
        logger.info("Loading model sam...")
        sam = sam_model_registry['vit_b']('pretrain/sam_vit_b_01ec64.pth')
        _ = sam.to(device='cuda')
        return sam


def loadimg(img_url,cfg):
    img = Image.open(img_url)
    #Change to tensor
    transform = transforms.Compose([transforms.Resize((480,480)),
                                    transforms.ToTensor()])
    img_t = transform(img)
    img = np.array(img)
    # print(img_t.size())
    return img_t,img,img.shape

def loadtxt(promote, word_length):
    txt = tokenize(promote, word_length, True).squeeze(0).unsqueeze(0)
    return txt,promote


def genrate_mask(img,txt,img_ori,cfg,aiunet = None,sam=None,rsam=None):
    if aiunet != None:
        img = img.to(cfg.device)
        txt = txt.to(cfg.device)
        pred = aiunet(img.unsqueeze(0),txt)
        # pred = torch.sigmoid(pred)
        # pred = pred.cpu().numpy()

        pred1 = pred > 0.5

        # return pred1
        return pred

    if sam != None:
        pred1 = genrate_mask(img,txt,img_ori,cfg,aiunet = load_models(cfg,'AIUNET_ORI'),sam=None)
        pred_mask = torch.tensor(pred1).to(cfg.device).float()
        pred_mask = F.interpolate(pred_mask.reshape(1,1,480,480), [256, 256], mode='bilinear')
        img = img.unsqueeze(0).to('cuda')
        img = F.interpolate(img, [1024, 1024], mode='bilinear')

        input = []
        img2 = cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB)

        input = []
        # point_coords = torch.tensor([[[240,240]]]).to(cfg.device)
        # point_labels = torch.tensor([[1]]).to(cfg.device)

        # input.append({'image': img, 'original_size': (480, 480), 'boxes': None, 'point_coords':point_coords,'point_labels':point_labels,'mask_inputs': pred_mask})
        input.append({'image': img.squeeze(0), 'original_size': (480, 480), 'mask_inputs': pred_mask})

        pred2 = sam(input,True)
        # generator = SamAutomaticMaskGenerator(sam, output_mode='binary_mask')
        # pred2 = generator.generate(img2,pred_mask)
        # pred2 = generator.generate(img2)
        pred2[0]['masks'] = pred2[0]['masks'][:,-1,:,:].squeeze().cpu().numpy()
        pred2[0]['masks'] = np.array(pred2[0]['masks'])
        count = 0
        for i in pred2[0]['masks'].flat:
            if i == True:
                count += 1
        print("     The pred vlaue is:" + str(count))

        return pred2[0]['masks']

    if rsam != None:
        # img = F.interpolate(img.unsqueeze(0),(1024,1024),mode='bilinear')
        img = img.unsqueeze(0)
        img = img.to(cfg.device)
        txt = txt.to(cfg.device)
        pred = rsam(img, txt)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy()

        pred3 = np.array(pred > 0.35)

        return pred3

def save_mask(root_path,img_name,img,mask,ori_size):
    img_name = '{}-pred.jpg'.format(img_name)

    # count = 0
    # for i in mask.flat:
    #     if i == False:
    #         count += 1
    # print(count)

    process = transforms.Compose([
        transforms.Resize((ori_size[0],ori_size[1]))
    ])
    mask = Image.fromarray(mask)
    mask = process(mask)
    mask = np.array(mask)
    mask = mask*255
    save_path = os.path.join(root_path, img_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if os.path.exists(save_path):
        os.remove(save_path)
    cv2.imwrite(filename=save_path,
                img=vis(img, mask))


if __name__ == '__main__':
    img_name = 'demo02.jpg'
    promote = 'zebra'
    # 'RSAM','AIUNET','SAM'
    model_name = 'RSAM'
    # model_name = 'AIUNET'
    # model_name = 'SAM'
    demo(img_name,promote,model_name)
