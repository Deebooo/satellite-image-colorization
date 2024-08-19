import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_sample_images(generator, fixed_grayscale, fixed_real_color, epoch, metrics, val_loss_G, val_loss_D):
    save_dir = '/local_disk/helios/skhelil/fichiers/Linkedin/'

    generator.eval()
    with torch.no_grad():
        gen_color = generator(fixed_grayscale)

    fixed_grayscale = (fixed_grayscale * 0.5 + 0.5).cpu().numpy()
    fixed_real_color = (fixed_real_color * 0.5 + 0.5).cpu().numpy()
    gen_color = (gen_color * 0.5 + 0.5).cpu().numpy()

    metrics_text = (f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
                    f"F1 Score: {metrics['f1']:.3f} | "
                    f"PSNR: {metrics['psnr']:.3f} | SSIM: {metrics['ssim']:.3f} | "
                    f"Val G loss: {val_loss_G:.3f} | Val D loss: {val_loss_D:.3f}")

    for i in range(3):  # Always plot 3 images
        sample_save_dir = os.path.join(save_dir, f'sample_{i}')

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Epoch {epoch}', fontsize=16)

        # Top row: Grayscale and Real color images
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(fixed_grayscale[i][0], cmap='gray')
        ax1.set_title('Input (Grayscale)')
        ax1.axis('off')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(np.transpose(fixed_real_color[i], (1, 2, 0)))
        ax2.set_title('Real (Color)')
        ax2.axis('off')

        # Bottom row: Generated color image
        ax3 = fig.add_subplot(2, 1, 2)
        ax3.imshow(np.transpose(gen_color[i], (1, 2, 0)))
        ax3.set_title('Generated (Color)')
        ax3.axis('off')

        # Add metrics text at the bottom
        fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12)

        plt.savefig(os.path.join(sample_save_dir, f'sample_image_epoch_{epoch}_sample_{i}.png'))
        plt.close()

    generator.train()