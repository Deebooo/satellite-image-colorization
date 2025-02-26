import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio
from utils.lab2rgb import lab2rgb
from skimage import color

def create_lab_image(L, AB):
    # Convert PyTorch tensors to NumPy arrays
    L = (L.cpu().numpy().astype(np.float32) + 1.0) * 50.0  # L back to [0, 100]
    AB = AB.cpu().numpy().astype(np.float32) * 128.0  # AB back to [-128, 128]

    # Initialize an empty array for the LAB image
    batch_size = L.shape[0]
    height = L.shape[2]
    width = L.shape[3]

    lab_image = np.zeros((batch_size, height, width, 3), dtype=np.float32)

    # Assign L to the first channel of the LAB image
    lab_image[:, :, :, 0] = L.squeeze(1)

    # Assign AB to the second and third channels of the LAB image
    lab_image[:, :, :, 1:] = np.transpose(AB, (0, 2, 3, 1))

    return lab_image

def calculate_metrics(generator, dataloader, device):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    psnr_values = []
    delta_e_values = []

    generator.eval()

    with torch.no_grad():
        for l_channel, real_ab in dataloader:
            l_channel = l_channel.to(device)
            real_ab = real_ab.to(device)

            gen_ab = generator(l_channel)

            # Convert LAB to RGB using the lab2rgb function
            real_rgb = lab2rgb(l_channel, real_ab).to(device)
            gen_rgb = lab2rgb(l_channel, gen_ab).to(device)

            # Calculate PSNR
            psnr_value = psnr(gen_rgb, real_rgb)

            psnr_values.append(psnr_value.item())

            # Create LAB images
            real_lab = create_lab_image(l_channel, real_ab)
            gen_lab = create_lab_image(l_channel, gen_ab)
            delta_e_value = np.mean([color.deltaE_ciede2000(real_lab[i], gen_lab[i]) for i in range(real_lab.shape[0])])
            delta_e_values.append(delta_e_value)

    # Compute mean metrics over the entire dataset
    final_psnr = torch.mean(torch.tensor(psnr_values)).item()
    final_delta_e = np.mean(delta_e_values)

    return {
        'psnr': final_psnr,
        'ciede2000': final_delta_e
    }