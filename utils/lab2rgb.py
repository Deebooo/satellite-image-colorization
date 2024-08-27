import torch
import numpy as np
from skimage import color

def lab2rgb(L, AB):
    """
    Convert LAB channels to RGB.
    L is expected to be in the range [-1, 1] (which maps to [0, 100] in LAB space).
    AB is expected to be in the range [-1, 1] (which maps to [-128, 128] in LAB space).
    The function outputs an RGB image in the range [0, 1].
    """
    L = (L + 1) * 50  # Map L from [-1, 1] to [0, 100]
    AB = AB * 128     # Map AB from [-1, 1] to [-128, 128]

    L = torch.clamp((L + 1) * 50, 0, 100)
    AB = torch.clamp(AB * 128, -128, 127)

    # Stack the channels to get a LAB image
    lab_image = torch.cat([L, AB], dim=1)

    # Convert LAB to RGB using skimage (expects numpy array)
    lab_image_np = lab_image.cpu().numpy().transpose((0, 2, 3, 1))  # Convert to (batch_size, height, width, channels)
    rgb_image_np = color.lab2rgb(lab_image_np)  # Convert to RGB

    # Convert back to torch tensor and reshape to match the original input dimensions
    rgb_image = torch.tensor(rgb_image_np).permute(0, 3, 1, 2).to(L.device)

    return rgb_image