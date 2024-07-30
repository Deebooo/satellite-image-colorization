import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_sample_images(generator, grayscale, real_color, epoch):
    save_dir = '/local_disk/helios/skhelil/fichiers/GAN_training/'

    generator.eval()
    with torch.no_grad():
        gen_color = generator(grayscale)

    grayscale = (grayscale * 0.5 + 0.5).cpu().numpy()
    real_color = (real_color * 0.5 + 0.5).cpu().numpy()
    gen_color = (gen_color * 0.5 + 0.5).cpu().numpy()

    for i in range(min(3, grayscale.shape[0])):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(grayscale[i][0], cmap='gray')
        ax1.set_title('Input (Grayscale)')
        ax2.imshow(np.transpose(real_color[i], (1, 2, 0)))
        ax2.set_title('Real (Color)')
        ax3.imshow(np.transpose(gen_color[i], (1, 2, 0)))
        ax3.set_title('Generated (Color)')
        plt.savefig(os.path.join(save_dir, f'sample_image_epoch_{epoch}_sample_{i}.png'))
        plt.close()

    generator.train()