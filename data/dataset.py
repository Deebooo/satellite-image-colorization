import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import torchvision.transforms as transforms
import cv2

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
            with rasterio.open(img_path) as src:
                image = src.read([1, 2, 3])
                image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
                lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_files))

        l_channel = lab_image[:, :, 0]
        ab_channels = lab_image[:, :, 1:]

        l_channel = Image.fromarray(l_channel)
        ab_channels = Image.fromarray(ab_channels)

        if self.transform:
            l_channel = self.transform(l_channel)
            ab_channels = self.transform(ab_channels)

        # Normalize L channel to [-1, 1] and AB channels to [-1, 1]
        l_channel = (l_channel / 50) - 1
        ab_channels = ab_channels / 128

        return l_channel.float(), ab_channels.float()