import torch
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
from utils.lab2rgb import lab2rgb

def calculate_metrics(generator, dataloader, device):
    ssim_scores = []
    psnr_values = []
    mse_values = []
    lpips_values = []

    # Initialize SSIM and LPIPS metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)

    generator.eval()

    with torch.no_grad():
        for l_channel, real_ab in dataloader:
            l_channel = l_channel.to(device)
            real_ab = real_ab.to(device)

            gen_ab = generator(l_channel)

            # Convert LAB to RGB using the lab2rgb function
            real_rgb = lab2rgb(l_channel, real_ab).to(device)
            gen_rgb = lab2rgb(l_channel, gen_ab).to(device)

            # Ensure the images are in the range [-1, 1] for LPIPS
            real_rgb = (real_rgb * 2) - 1
            gen_rgb = (gen_rgb * 2) - 1

            # Calculate metrics using RGB images
            for i in range(real_rgb.shape[0]):
                # SSIM Calculation
                ssim = ssim_metric(gen_rgb[i].unsqueeze(0), real_rgb[i].unsqueeze(0))
                ssim_scores.append(ssim.item())

                # MSE Calculation
                mse = F.mse_loss(gen_rgb[i], real_rgb[i]).item()
                mse_values.append(mse)

                # PSNR Calculation
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                psnr_values.append(psnr)

                # LPIPS Calculation
                lpips_score = lpips_metric(gen_rgb[i].unsqueeze(0), real_rgb[i].unsqueeze(0))
                lpips_values.append(lpips_score.item())

    # Calculate average metrics over the entire dataset
    final_ssim = np.mean(ssim_scores)
    final_psnr = np.mean(psnr_values)
    final_mse = np.mean(mse_values)
    final_lpips = np.mean(lpips_values)

    return {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'mse': final_mse,
        'lpips': final_lpips,
    }