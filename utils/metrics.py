import torch
import numpy as np
from torchmetrics.functional import precision as precision_metric
from torchmetrics.functional import recall as recall_metric
from torchmetrics.functional import f1_score as f1_score_metric
from torchmetrics.functional import accuracy as accuracy_metric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import torch.nn.functional as F

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
        for grayscale, real_color in tqdm(dataloader):
            grayscale = grayscale.to(device)
            real_color = real_color.to(device)

            gen_color = generator(grayscale)

            # Calculate pixel-wise metrics for the entire batch on GPU
            real_binary = (real_color > 0.5).float()
            gen_binary = (gen_color > 0.5).float()

            # Calculate precision, recall, F1 score using torchmetrics
            precision = precision_metric(gen_binary, real_binary, task='binary').to(device)
            recall = recall_metric(gen_binary, real_binary, task='binary').to(device)
            f1 = f1_score_metric(gen_binary, real_binary, task="binary").to(device)
            accuracy = accuracy_metric(gen_binary, real_binary, task="binary").to(device)

            # Calculate SSIM using torchmetrics
            ssim_score = ssim(gen_color, real_color)

            precisions.append(precision.item())
            recalls.append(recall.item())
            f1s.append(f1.item())
            accuracies.append(accuracy.item())
            ssim_scores.append(ssim_score.item())

            # Calculate PSNR for the entire batch
            mse = F.mse_loss(gen_color, real_color).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)

    # Compute final SSIM
    final_ssim = ssim.compute()

    return {
        'precision': torch.mean(torch.tensor(precisions)).item(),
        'recall': torch.mean(torch.tensor(recalls)).item(),
        'f1': torch.mean(torch.tensor(f1s)).item(),
        'psnr': np.mean(psnr_values),
        'ssim': final_ssim.item(),
    }