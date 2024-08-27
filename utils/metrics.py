import torch
import numpy as np
from torchmetrics.functional import precision as precision_metric
from torchmetrics.functional import recall as recall_metric
from torchmetrics.functional import f1_score as f1_score_metric
from torchmetrics.functional import accuracy as accuracy_metric
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import cv2

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

            # Convert LAB to RGB for metric calculation
            l_channel_np = l_channel.cpu().numpy() * 50.0 + 50.0
            real_ab_np = real_ab.cpu().numpy() * 128.0
            gen_ab_np = gen_ab.cpu().numpy() * 128.0

            """
            # Convert LAB to RGB for metric calculation
            l_channel_np = l_channel.cpu().numpy() * 50.0  # Rescale L channel back to [0, 100]
            real_ab_np = (real_ab.cpu().numpy() + 1.0) * 128.0  # Rescale AB channels back to [-128, 127]
            gen_ab_np = (gen_ab.cpu().numpy() + 1.0) * 128.0
            """

            real_lab = np.concatenate([l_channel_np, real_ab_np], axis=1)
            gen_lab = np.concatenate([l_channel_np, gen_ab_np], axis=1)

            real_rgb = cv2.cvtColor(real_lab.transpose(1, 2, 0), cv2.COLOR_LAB2RGB)
            gen_rgb = cv2.cvtColor(gen_lab.transpose(1, 2, 0), cv2.COLOR_LAB2RGB)

            real_rgb_tensor = torch.from_numpy(real_rgb).permute(2, 0, 1).float().to(device)
            gen_rgb_tensor = torch.from_numpy(gen_rgb).permute(2, 0, 1).float().to(device)

            # Calculate metrics as before but now using the RGB tensors
            real_binary = (real_rgb_tensor > 0.5).float()
            gen_binary = (gen_rgb_tensor > 0.5).float()

            precision = precision_metric(gen_binary, real_binary, task='binary').to(device)
            recall = recall_metric(gen_binary, real_binary, task='binary').to(device)
            f1 = f1_score_metric(gen_binary, real_binary, task="binary").to(device)
            accuracy = accuracy_metric(gen_binary, real_binary, task="binary").to(device)

            ssim_score = ssim(gen_rgb_tensor, real_rgb_tensor)

            precisions.append(precision.item())
            recalls.append(recall.item())
            f1s.append(f1.item())
            accuracies.append(accuracy.item())
            ssim_scores.append(ssim_score.item())

            mse = F.mse_loss(gen_rgb_tensor, real_rgb_tensor).item()
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