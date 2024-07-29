import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
from tqdm import tqdm

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

            real_np = real_color.cpu().numpy()
            gen_np = gen_color.cpu().numpy()

            real_binary = (real_np > 0.5).astype(int)
            gen_binary = (gen_np > 0.5).astype(int)

            precision = precision_score(real_binary.flatten(), gen_binary.flatten(), average='binary', zero_division=1)
            recall = recall_score(real_binary.flatten(), gen_binary.flatten(), average='binary', zero_division=1)
            f1 = f1_score(real_binary.flatten(), gen_binary.flatten(), average='binary', zero_division=1)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            mse = np.mean((real_np - gen_np) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)

            for i in range(gen_np.shape[0]):
                ssim_score = ssim(real_np[i].transpose(1, 2, 0),
                                  gen_np[i].transpose(1, 2, 0),
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