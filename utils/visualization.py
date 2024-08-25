import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def save_sample_images(generator, fixed_grayscale, fixed_real_ab, epoch, metrics, val_loss_G, val_loss_D):
    save_dir = '/local_disk/helios/skhelil/fichiers/Linkedin/'

    generator.eval()
    with torch.no_grad():
        gen_ab = generator(fixed_grayscale)

    fixed_grayscale_np = (fixed_grayscale * 0.5 + 0.5).cpu().numpy().astype(np.uint8)
    fixed_real_ab_np = (fixed_real_ab * 0.5 + 0.5).cpu().numpy().astype(np.uint8)
    gen_ab_np = (gen_ab * 0.5 + 0.5).cpu().numpy().astype(np.uint8)

    for i in range(3):
        sample_save_dir = os.path.join(save_dir, f'sample_{i}')

        real_lab = np.concatenate([fixed_grayscale_np[i][0], fixed_real_ab_np[i]], axis=0)
        gen_lab = np.concatenate([fixed_grayscale_np[i][0], gen_ab_np[i]], axis=0)

        real_rgb = cv2.cvtColor(real_lab, cv2.COLOR_LAB2RGB)
        gen_rgb = cv2.cvtColor(gen_lab, cv2.COLOR_LAB2RGB)

        metrics_text = (f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
                        f"F1 Score: {metrics['f1']:.3f} | "
                        f"PSNR: {metrics['psnr']:.3f} | SSIM: {metrics['ssim']:.3f} | "
                        f"Val G loss: {val_loss_G:.3f} | Val D loss: {val_loss_D:.3f}")

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Epoch {epoch}', fontsize=16)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(fixed_grayscale_np[i][0], cmap='gray')
        ax1.set_title('Input (L channel)')
        ax1.axis('off')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(real_rgb)
        ax2.set_title('Real (Color)')
        ax2.axis('off')

        ax3 = fig.add_subplot(2, 1, 2)
        ax3.imshow(gen_rgb)
        ax3.set_title('Generated (Color)')
        ax3.axis('off')

        fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12)

        plt.savefig(os.path.join(sample_save_dir, f'sample_image_epoch_{epoch}_sample_{i}.png'))
        plt.close()

    generator.train()