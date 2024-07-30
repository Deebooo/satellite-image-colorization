import torch
import numpy as np
from torchmetrics.functional import precision as precision_metric
from torchmetrics.functional import recall as recall_metric
from torchmetrics.functional import f1_score as f1_score_metric
from torchmetrics.functional import jaccard_index as jaccard_metric
from torchmetrics.functional import accuracy as accuracy_metric
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import torch.nn.functional as F

def calculate_metrics(generator, dataloader, device):
    precisions = []
    recalls = []
    f1s = []
    ssim_scores = []
    psnr_values = []
    jaccard_scores = []
    accuracies = []

    generator.eval()

    with torch.no_grad():
        for grayscale, real_color in tqdm(dataloader):
            grayscale = grayscale.to(device)
            real_color = real_color.to(device)

            gen_color = generator(grayscale)

            # Calculate pixel-wise metrics for the entire batch on GPU
            real_binary = (real_color > 0.5).float()
            gen_binary = (gen_color > 0.5).float()

            # Calculate precision, recall, F1 score, Jaccard index, and accuracy using torchmetrics
            precision = precision_metric(gen_binary, real_binary, task='binary').to(device)
            recall = recall_metric(gen_binary, real_binary, task='binary').to(device)
            f1 = f1_score_metric(gen_binary, real_binary, task="binary").to(device)
            jaccard = jaccard_metric(gen_binary, real_binary, task="binary").to(device)
            accuracy = accuracy_metric(gen_binary, real_binary, task="binary").to(device)

            # Calculate SSIM for each image in the batch /!\
            real_color_np = real_color.cpu().numpy()
            gen_color_np = gen_color.cpu().numpy()
            for i in range(gen_color_np.shape[0]):
                ssim_score = ssim(real_color_np[i].transpose(1, 2, 0),
                                  gen_color_np[i].transpose(1, 2, 0),
                                  multichannel=True, data_range=gen_color_np[i].max() - gen_color_np[i].min(),
                                  win_size=11, channel_axis=2)
                ssim_scores.append(ssim_score)

            precisions.append(precision.item())
            recalls.append(recall.item())
            f1s.append(f1.item())
            jaccard_scores.append(jaccard.item())
            accuracies.append(accuracy.item())
            ssim_scores.append(ssim)

            # Calculate PSNR for the entire batch
            mse = F.mse_loss(gen_color, real_color).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)

    return {
        'precision': torch.mean(torch.tensor(precisions)).item(),
        'recall': torch.mean(torch.tensor(recalls)).item(),
        'f1': torch.mean(torch.tensor(f1s)).item(),
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_scores),
        'jaccard': torch.mean(torch.tensor(jaccard_scores)).item()
    }