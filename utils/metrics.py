import torch
import numpy as np
from torchmetrics.functional import precision as precision_metric
from torchmetrics.functional import recall as recall_metric
from torchmetrics.functional import f1_score as f1_score_metric
from torchmetrics.functional import accuracy as accuracy_metric
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
from utils.lab2rgb import lab2rgb


def calculate_metrics(generator, dataloader, device):
    precisions = []
    recalls = []
    f1s = []
    ssim_scores = []
    psnr_values = []
    accuracies = []

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    generator.eval()

    with torch.no_grad():
        for l_channel, real_ab in dataloader:
            l_channel = l_channel.to(device)
            real_ab = real_ab.to(device)

            gen_ab = generator(l_channel)

            # Convert LAB to RGB using the lab2rgb function
            real_rgb = lab2rgb(l_channel, real_ab)
            gen_rgb = lab2rgb(l_channel, gen_ab)

            # Calculate metrics using RGB images
            real_binary = (real_rgb > 0.5).float()
            gen_binary = (gen_rgb > 0.5).float()

            precision = precision_metric(gen_binary, real_binary, task='binary').to(device)
            recall = recall_metric(gen_binary, real_binary, task='binary').to(device)
            f1 = f1_score_metric(gen_binary, real_binary, task="binary").to(device)
            accuracy = accuracy_metric(gen_binary, real_binary, task="binary").to(device)

            ssim_score = ssim(gen_rgb, real_rgb)

            precisions.append(precision.item())
            recalls.append(recall.item())
            f1s.append(f1.item())
            accuracies.append(accuracy.item())
            ssim_scores.append(ssim_score.item())

            mse = F.mse_loss(gen_rgb, real_rgb).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)

    final_ssim = ssim.compute()

    return {
        'precision': torch.mean(torch.tensor(precisions)).item(),
        'recall': torch.mean(torch.tensor(recalls)).item(),
        'f1': torch.mean(torch.tensor(f1s)).item(),
        'psnr': np.mean(psnr_values),
        'ssim': final_ssim.item(),
    }
