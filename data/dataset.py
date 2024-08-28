import rasterio
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from skimage import io, color

class SatelliteImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            rgb_image = io.imread(img_path) # Load the .tif image using skimage
            '''
            with rasterio.open(img_path) as src:
                rgb_image = src.read([1, 2, 3])
                rgb_image = np.transpose(rgb_image, (1, 2, 0)).astype(np.float32) # Load the .tif image using rasterio`
            '''
            # Ensure image is in float format and normalized to [0, 1]
            rgb_image = rgb_image.astype(np.float32) / 255.0

            # Convert the RGB image to LAB
            lab_image = color.rgb2lab(rgb_image)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_files))

        # Extract L and AB channels
        l_channel = lab_image[:, :, 0]
        ab_channels = lab_image[:, :, 1:]

        # Normalize L to [-1, 1] and AB to [-1, 1]
        l_channel = (l_channel / 50.0) - 1.0
        ab_channels = ab_channels / 128.0

        # Apply transformations if provided
        if self.transform:
            l_channel = self.transform(l_channel)
            ab_channels = self.transform(ab_channels)

        # Convert to PyTorch tensors and add channel dimensions
        l_channel = torch.from_numpy(l_channel).unsqueeze(0)
        ab_channels = torch.from_numpy(ab_channels).permute(2, 0, 1)

        return l_channel.float(), ab_channels.float()


