import torch
from torch.utils.data import DataLoader, random_split
from data.dataset import SatelliteImageDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.utils import weights_init_normal
from train import train
from torchvision import transforms
import os
from utils.resource_monitor import ResourceMonitor
import time

if __name__ == "__main__":
    batch_size = 64
    val_batch_size = 64
    num_epochs = 300
    image_size = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Set up resource monitoring
    log_directory = "/home/nas-wks01/users/uapv2300011/gan/results/log"
    os.makedirs(log_directory, exist_ok=True)

    monitor = ResourceMonitor(log_directory)
    monitor.start()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size))
    ])

    dataset = SatelliteImageDataset("/home/nas-wks01/users/uapv2300011/gan/datas/tiles", transform=transform)

    # Print dataset info for verification
    print(f"Dataset length: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    generator = Generator()
    discriminator = Discriminator()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    try:
        trained_generator, trained_discriminator = train(generator, discriminator, train_dataloader, val_dataloader,
                                                         num_epochs, device)
        print(f"Learning Rate for Generator: {lr_G}")
        print(f"Learning Rate for Discriminator: {lr_D}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Stop the monitoring
        monitor.stop()

    print("Training complete.")