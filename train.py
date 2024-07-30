import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import calculate_metrics
from utils.visualization import save_sample_images

def validate(generator, discriminator, dataloader, criterion_GAN, criterion_pixelwise, device, lambda_pixel):
    generator.eval()
    discriminator.eval()
    total_loss_G = 0

    with torch.no_grad():
        for grayscale, real_color in dataloader:
            grayscale = grayscale.to(device)
            real_color = real_color.to(device)
            valid = torch.ones((grayscale.size(0), 1, 15, 15), requires_grad=False).to(device)

            gen_color = generator(grayscale)
            pred_fake = discriminator(grayscale, gen_color)

            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(gen_color, real_color)
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            total_loss_G += loss_G.item()

    metrics = calculate_metrics(generator, dataloader, device)

    generator.train()
    discriminator.train()

    return total_loss_G / len(dataloader), metrics


def train(generator, discriminator, train_dataloader, val_dataloader, num_epochs, device):
    generator.to(device)
    discriminator.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism for generator.")
        generator = nn.DataParallel(generator)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism for discriminator.")
        discriminator = nn.DataParallel(discriminator)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()

    lambda_pixel = 100
    early_stopping_patience = 10
    no_improve_epochs = 0

    best_loss = float('inf')
    best_ssim = float('-inf')

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        total_loss_G = 0
        total_loss_D = 0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            for grayscale, real_color in tepoch:
                grayscale = grayscale.to(device)
                real_color = real_color.to(device)

                valid = torch.ones((grayscale.size(0), 1, 15, 15), requires_grad=False).to(device)
                fake = torch.zeros((grayscale.size(0), 1, 15, 15), requires_grad=False).to(device)

                optimizer_G.zero_grad()
                gen_color = generator(grayscale)
                pred_fake = discriminator(grayscale, gen_color)
                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_pixel = criterion_pixelwise(gen_color, real_color)
                loss_G = loss_GAN + lambda_pixel * loss_pixel
                loss_G.backward()

                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
                optimizer_G.step()

                optimizer_D.zero_grad()
                pred_real = discriminator(grayscale, real_color)
                loss_real = criterion_GAN(pred_real, valid)
                pred_fake = discriminator(grayscale, gen_color.detach())
                loss_fake = criterion_GAN(pred_fake, fake)
                loss_D = 0.5 * (loss_real + loss_fake)
                loss_D.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                optimizer_D.step()

                total_loss_G += loss_G.item()
                total_loss_D += loss_D.item()

                tepoch.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())

        avg_loss_G = total_loss_G / len(train_dataloader)
        avg_loss_D = total_loss_D / len(train_dataloader)

        val_loss_G, metrics = validate(generator, discriminator, val_dataloader, criterion_GAN, criterion_pixelwise,
                                       device, lambda_pixel)

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"[D loss: {avg_loss_D:.3f}] [G loss: {avg_loss_G:.3f}] "
              f"[Val G loss: {val_loss_G:.3f}] "
              f"[Precision: {metrics['precision']:.3f}] "
              f"[Recall: {metrics['recall']:.3f}] [F1 Score: {metrics['f1']:.3f}] "
              f"[Jaccard: {metrics['jaccard']:.3f}] "
              f"[PSNR: {metrics['psnr']:.3f}] [SSIM: {metrics['ssim']:.3f}] ")


        if val_loss_G < best_loss and metrics['ssim'] > best_ssim:
            best_loss = val_loss_G
            best_ssim = metrics['ssim']
            no_improve_epochs = 0
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss_G': avg_loss_G,
                'loss_D': avg_loss_D,
                'ssim': metrics['ssim'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'psnr': metrics['psnr'],
                'jaccard': metrics['jaccard']
            }, 'best_checkpoint.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break

        if epoch % 5 == 0:
            save_sample_images(generator, grayscale, real_color, epoch)

    return generator, discriminator