import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data.dataset import SatelliteImageDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.utils import weights_init_normal
from train import train

if __name__ == "__main__":
    batch_size = 64
    val_batch_size = 64
    num_epochs = 50
    image_size = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = SatelliteImageDataset("/local_disk/helios/skhelil/fichiers/images_satt/tiles", transform=transform)

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
    except Exception as e:
        print(f"An error occurred during training: {e}")

    print("Training complete.")