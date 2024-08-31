import os
import matplotlib.pyplot as plt
import torch
from utils.lab2rgb import lab2rgb

def save_sample_images(generator, fixed_l_channel, fixed_real_ab, epoch, metrics, val_loss_G, val_loss_D):
    save_dir = '/local_disk/helios/skhelil/fichiers/Linkedin/'

    generator.eval()
    with torch.no_grad():
        gen_ab = generator(fixed_l_channel)

    # Convert LAB to RGB using the lab2rgb function
    fixed_real_rgb = lab2rgb(fixed_l_channel, fixed_real_ab)
    gen_rgb = lab2rgb(fixed_l_channel, gen_ab)

    for i in range(3):
        sample_save_dir = os.path.join(save_dir, f'sample_{i}')

        metrics_text = (f"PSNR: {metrics['psnr']:.3f} | SSIM: {metrics['ssim']:.3f} | "
                        f"CIEDE2000: {metrics['ciede2000']:.3f} | "
                        f"Val G loss: {val_loss_G:.3f} | Val D loss: {val_loss_D:.3f}")

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Epoch {epoch}', fontsize=16)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(fixed_l_channel[i][0].cpu().numpy() * 50 + 50, cmap='gray')
        ax1.set_title('Input (L channel)')
        ax1.axis('off')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(fixed_real_rgb[i].squeeze(0).permute(1, 2, 0).cpu().numpy())
        ax2.set_title('Real (Color)')
        ax2.axis('off')

        ax3 = fig.add_subplot(2, 1, 2)
        ax3.imshow(gen_rgb[i].squeeze(0).permute(1, 2, 0).cpu().numpy())
        ax3.set_title('Generated (Color)')
        ax3.axis('off')

        fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12)

        plt.savefig(os.path.join(sample_save_dir, f'sample_image_epoch_{epoch}_sample_{i}.png'))
        plt.close()

    generator.train()