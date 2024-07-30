import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import precision as precision_metric
from torchmetrics.functional import recall as recall_metric
from tqdm import tqdm
import torch.nn.functional as F

def calculate_metrics(generator, dataloader, device):
    precisions = []
    recalls = []
    f1s = []
    ssim_scores = []
    psnr_values = []

    generator.eval()

    with torch.no_grad():
        for grayscale, real_color in tqdm(dataloader):
            grayscale = grayscale.to(device)
            real_color = real_color.to(device)

            gen_color = generator(grayscale)

            # Calculate pixel-wise metrics for the entire batch on GPU
            real_binary = (real_color > 0.5).float()
            gen_binary = (gen_color > 0.5).float()

            # Calculate precision, recall, and F1 score using torchmetrics
            precision = precision_metric(gen_binary, real_binary, task='binary').to(device)
            recall = recall_metric(gen_binary, real_binary, task='binary').to(device)

            # Calculate F1 score using the previously calculated precision and recall
            if precision.item() + recall.item() > 0:
                f1 = 2 * (precision.item() * recall.item()) / (precision + recall.item())
            else:
                f1 = 0.0

            precisions.append(precision.item())
            recalls.append(recall.item())
            f1s.append(f1)

            # Calculate PSNR for the entire batch
            mse = F.mse_loss(gen_color, real_color).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)

            # Calculate SSIM for each image in the batch
            real_color_np = real_color.cpu().numpy()
            gen_color_np = gen_color.cpu().numpy()
            for i in range(gen_color_np.shape[0]):
                ssim_score = ssim(real_color_np[i].transpose(1, 2, 0),
                                  gen_color_np[i].transpose(1, 2, 0),
                                  multichannel=True, data_range=gen_color_np[i].max() - gen_color_np[i].min(),
                                  win_size=7, channel_axis=2)
                ssim_scores.append(ssim_score)

    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s),
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_scores)
    }