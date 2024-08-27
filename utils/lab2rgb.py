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
    AB2 = AB * 110.0
    L2 = (L + 1.0) * 50.0
    Lab = torch.cat([L2, AB2], dim=1)
    Lab = Lab[0].data.cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
    rgb = color.lab2rgb(Lab) * 255
    return rgb
