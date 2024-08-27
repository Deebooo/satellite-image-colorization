import numpy as np
from skimage import color
import torch

def lab2rgb(L, AB):
    """
    Convert a LAB tensor image to an RGB numpy output.

    Parameters:
    - L: (1-channel tensor array) L channel images (range: [-1, 1], torch tensor array)
    - AB: (2-channel tensor array) AB channel images (range: [-1, 1], torch tensor array)

    Returns:
    - rgb: (RGB numpy image) rgb output images (range: [0, 255], numpy array)
    """
    AB2 = AB * 128.0
    L2 = (L + 1.0) * 50.0

    # Concatenate L and AB channels
    Lab = torch.cat([L2, AB2], dim=1).cpu().float().numpy()

    # Initialize a list to store converted images
    rgb_images = []

    # Convert each image in the batch
    for i in range(Lab.shape[0]):
        Lab_image = np.transpose(Lab[i], (1, 2, 0)).astype(np.float64)
        rgb_image = color.lab2rgb(Lab_image) * 255
        rgb_images.append(rgb_image.astype(np.uint8))

    # Stack and return as numpy array
    rgb_images = np.stack(rgb_images, axis=0)
    return rgb_images
