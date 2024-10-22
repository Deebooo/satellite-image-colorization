import os
import torch
import torch.nn as nn
import numpy as np
import rasterio
from rasterio.windows import Window
from utils.lab2rgb import lab2rgb
from skimage import color


# Generator class (same as in the training script)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def down_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, dropout=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.down1 = down_block(1, 64, normalize=False) # The input has 1 channel (L channel)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, normalize=False)

        self.up1 = up_block(512, 512, dropout=True)
        self.up2 = up_block(1024, 512, dropout=True)
        self.up3 = up_block(1024, 512, dropout=True)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1),  # Output 2 channels (A and B)
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))

        return self.up8(torch.cat([u7, d1], 1))


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator().to(device)

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)

    # Check if the state dict was saved from a DataParallel model
    if list(state_dict.keys())[0].startswith('module.'):
        # Create a new OrderedDict without the 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v

        # Load the modified state dict
        model.load_state_dict(new_state_dict)
    else:
        # Load the state dict as is
        model.load_state_dict(state_dict)

    model.eval()
    return model, device


def preprocess_tile(tile):
    # Convert the tile to float and normalize to [0, 1]
    tile = tile.astype(np.float32) / 255.0

    # Convert the RGB image to LAB
    lab_tile = color.rgb2lab(tile)

    # Extract L and AB channels
    l_channel = lab_tile[:, :, 0]
    ab_channels = lab_tile[:, :, 1:]

    # Normalize L to [-1, 1] and AB to [-1, 1]
    l_channel = (l_channel / 50.0) - 1.0
    ab_channels = ab_channels / 128.0

    # Convert to PyTorch tensors and add channel dimensions
    l_channel = torch.from_numpy(l_channel).unsqueeze(0).unsqueeze(0)
    ab_channels = torch.from_numpy(ab_channels).permute(2, 0, 1).unsqueeze(0)

    return l_channel, ab_channels


def predict_tile(model, l_channel, device):
    l_channel = l_channel.to(device)

    with torch.no_grad():
        output = model(l_channel)

    return output


def process_geotiff(input_path, output_path, model, device, tile_size=256):
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(count=3, dtype=rasterio.uint8)

        # Read the input image
        input_array = src.read()
        height, width = input_array.shape[1], input_array.shape[2]

        # Create the output image
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(0, height, tile_size):
                for j in range(0, width, tile_size):
                    window = Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
                    tile = src.read(window=window)

                    # Ensure the tile has 3 channels (RGB)
                    if tile.shape[0] == 1:
                        tile = np.repeat(tile, 3, axis=0)

                    tile = np.transpose(tile, (1, 2, 0))

                    # Pad the tile if it's smaller than tile_size
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        padded_tile = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                        padded_tile[:tile.shape[0], :tile.shape[1], :] = tile
                        tile = padded_tile

                    # Preprocess the tile
                    l_channel, _ = preprocess_tile(tile)

                    # Predict AB channels
                    gen_ab = predict_tile(model, l_channel, device)

                    # Convert LAB to RGB
                    colorized_tile = lab2rgb(l_channel, gen_ab)

                    # Convert to uint8 and ensure it's in the correct shape
                    colorized_tile = (colorized_tile.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    # Write the colorized tile to the output image, avoiding the padded pixels
                    tile_height, tile_width = min(tile_size, height - i), min(tile_size, width - j)
                    dst.write(colorized_tile[:tile_height, :tile_width, :].transpose(2, 0, 1), window=window)

    print(f"Colorized image saved to {output_path}")

def process_folder(input_folder, output_folder, model_path, tile_size=256):
    model, device = load_model(model_path)

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_rgb.tif")
            process_geotiff(input_path, output_path, model, device, tile_size)

if __name__ == "__main__":
    input_folder = "/home/nas-wks01/users/uapv2300011/gan/datas/predict_dataset/Paris"
    output_folder = "/home/nas-wks01/users/uapv2300011/gan/datas/predicted_dataset/Paris"
    model_path = "/home/nas-wks01/users/uapv2300011/gan/project/bon_v1/prime_generator.pth"
    process_folder(input_folder, output_folder, model_path)