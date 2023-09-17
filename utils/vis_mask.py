import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

def vis(img_file, mask_file):
    # 将分割图和原图合在一起
    if len(mask_file.shape) == 2:
        mask_file = np.expand_dims(mask_file,2)
    alpha = 0.5
    # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
    out = img_file * (1 - alpha) + mask_file * alpha
    return out


if __name__ == '__main__':

    imgfile = '25-img.jpg'
    pngfile = '25-iou=84.66-black_car.png'
    outpath = 'test.png'

    image = vis(imgfile,pngfile)
    image.save(outpath)
    image.show()