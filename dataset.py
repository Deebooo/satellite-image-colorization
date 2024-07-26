import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import torchvision.transforms as transforms

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
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_files))

        rgb_image = Image.fromarray(image)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        grayscale = transforms.Grayscale()(rgb_image)
        return grayscale, rgb_image