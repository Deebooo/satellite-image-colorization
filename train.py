import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import calculate_metrics
from utils.visualization import save_sample_images
from torch.optim.lr_scheduler import ReduceLROnPlateau

def validate(generator, discriminator, dataloader, criterion_GAN, criterion_pixelwise, device, lambda_pixel):
    generator.eval()
    discriminator.eval()
    total_loss_G = 0
    total_loss_D = 0

    with torch.no_grad():
        for grayscale, real_color in dataloader:
            grayscale = grayscale.to(device)
            real_color = real_color.to(device)
            valid = torch.ones((grayscale.size(0), 1, 15, 15), requires_grad=False).to(device)
            fake = torch.zeros((grayscale.size(0), 1, 15, 15), requires_grad=False).to(device)

            gen_color = generator(grayscale)
            pred_fake = discriminator(grayscale, gen_color)
            pred_real = discriminator(grayscale, real_color)

            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(gen_color, real_color)
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_real = criterion_GAN(pred_real, valid)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()

    metrics = calculate_metrics(generator, dataloader, device)

    generator.train()
    discriminator.train()

    return total_loss_G / len(dataloader), total_loss_D / len(dataloader), metrics


def train(generator, discriminator, train_dataloader, val_dataloader, num_epochs, device):
    generator.to(device)
    discriminator.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism for generator.")
        generator = nn.DataParallel(generator)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism for discriminator.")
        discriminator = nn.DataParallel(discriminator)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.00025, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.75, patience=5)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.75, patience=5)

    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()

    lambda_pixel = 85
    early_stopping_patience = 10
    no_improve_epochs = 0

    best_composite_score = float('-inf')

    # Randomly select 3 fixed images for plotting
    fixed_images = next(iter(val_dataloader))
    fixed_grayscale, fixed_real_color = fixed_images
    indices = random.sample(range(fixed_grayscale.size(0)), 3)
    fixed_grayscale = fixed_grayscale[indices].to(device)
    fixed_real_color = fixed_real_color[indices].to(device)

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

        val_loss_G, val_loss_D, metrics = validate(generator, discriminator, val_dataloader, criterion_GAN, criterion_pixelwise,
                                       device, lambda_pixel)

        # Update schedulers
        scheduler_G.step(val_loss_G)
        scheduler_D.step(val_loss_D)
        current_lr_G = scheduler_G.get_last_lr()[0]
        current_lr_D = scheduler_D.get_last_lr()[0]

        print(f"[Epoch {epoch}/{num_epochs}] with lr_G={current_lr_G} & lr_D={current_lr_D}, "
              f"[D loss: {avg_loss_D:.3f}] [G loss: {avg_loss_G:.3f}] "
              f"[Val G loss: {val_loss_G:.3f}] [Val D loss: {val_loss_D:.3f}]"
              f"[Precision: {metrics['precision']:.3f}] "
              f"[Recall: {metrics['recall']:.3f}] [F1 Score: {metrics['f1']:.3f}] "
              f"[PSNR: {metrics['psnr']:.3f}] [SSIM: {metrics['ssim']:.3f}] ")

        # Calculate composite score
        composite_score = 1 / (val_loss_G + 1e-8)

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            no_improve_epochs = 0
            torch.save(generator.state_dict(), 'prime_generator.pth')
            torch.save(discriminator.state_dict(), 'prime_discriminator.pth')
            print(f"[---> Prime models at Epoch {epoch}/{num_epochs}] "
                  f"[D loss: {avg_loss_D:.3f}] [G loss: {avg_loss_G:.3f}] "
                  f"[Val G loss: {val_loss_G:.3f}] [Val D loss: {val_loss_D:.3f}]"
                  f"[Precision: {metrics['precision']:.3f}] "
                  f"[Recall: {metrics['recall']:.3f}] [F1 Score: {metrics['f1']:.3f}] "
                  f"[PSNR: {metrics['psnr']:.3f}] [SSIM: {metrics['ssim']:.3f}] ")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break

        if epoch % 1 == 0:
            save_sample_images(generator, fixed_grayscale, fixed_real_color, epoch, metrics, val_loss_G, val_loss_D)

    return generator, discriminator