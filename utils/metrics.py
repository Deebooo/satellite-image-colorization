import torch
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
from utils.lab2rgb import lab2rgb


def calculate_metrics(generator, dataloader, device):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ssim_scores = []
    psnr_values = []
    mse_values = []

    generator.eval()

    with torch.no_grad():
        for l_channel, real_ab in dataloader:
            l_channel = l_channel.to(device)
            real_ab = real_ab.to(device)

            gen_ab = generator(l_channel)

            # Convert LAB to RGB using the lab2rgb function
            real_rgb = lab2rgb(l_channel, real_ab).to(device)
            gen_rgb = lab2rgb(l_channel, gen_ab).to(device)

            # SSIM Calculation for the batch
            ssim_batch = ssim_metric(gen_rgb, real_rgb)
            ssim_scores.append(ssim_batch.item())

            # MSE Calculation for the batch
            mse_batch = F.mse_loss(gen_rgb, real_rgb, reduction='mean').item()
            mse_values.append(mse_batch)

            # PSNR Calculation for the batch
            psnr_batch = 20 * np.log10(1.0 / np.sqrt(mse_batch))
            psnr_values.append(psnr_batch)

    # Calculate average metrics over the entire dataset
    final_ssim = np.mean(ssim_scores)
    final_psnr = np.mean(psnr_values)
    final_mse = np.mean(mse_values)

    return {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'mse': final_mse,
    }