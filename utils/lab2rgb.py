import numpy as np
from skimage import color
import torch

def lab2rgb(L, AB):
    """
    Convert LAB tensor image to RGB numpy array
    """
    L = (L + 1.0) * 50.0
    AB = AB * 128.0
    Lab = torch.cat([L, AB], dim=1).cpu().float().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = (color.lab2rgb(img.transpose(1, 2, 0)) * 255).astype(np.uint8)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
