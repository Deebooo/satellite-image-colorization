import torch
import numpy as np
import cv2


def lab2rgb(L, AB):
    # Denormalize L channel (which was scaled to [-1, 1] and originally in range [0, 100])
    L = (L + 1) * 50.0  # Now L should be in [0, 100]

    # Denormalize AB channels (which were scaled to [-1, 1] and originally in range [-128, 127])
    AB = AB * 128.0  # Now AB should be in [-128, 127]

    # Convert PyTorch tensors to NumPy arrays
    L = L.cpu().numpy().astype(np.float32)
    AB = AB.cpu().numpy().astype(np.float32)

    # Initialize an empty array for the LAB image
    batch_size = L.shape[0]
    height = L.shape[1]
    width = L.shape[2]

    lab_image = np.zeros((batch_size, height, width, 3), dtype=np.float32)

    # Assign L to the first channel of the LAB image
    lab_image[:, :, :, 0] = L

    # Assign AB to the second and third channels of the LAB image
    lab_image[:, :, :, 1] = AB[:, 0, :, :]  # A channel
    lab_image[:, :, :, 2] = AB[:, 1, :, :]  # B channel

    # Convert LAB image to RGB using OpenCV in a loop for each image in the batch
    rgb_images = []
    for i in range(batch_size):
        rgb_image = cv2.cvtColor(lab_image[i], cv2.COLOR_LAB2RGB)
        rgb_images.append(rgb_image)

    # Stack the list of RGB images into a single NumPy array and convert to torch tensor
    rgb_images = np.stack(rgb_images, axis=0)

    # Convert to PyTorch tensor and normalize to [0, 1] if needed
    rgb_images = torch.from_numpy(rgb_images).permute(0, 3, 1, 2)  # Change to (batch_size, C, H, W)

    return rgb_images.float() / 255.0  # Return a float tensor in the range [0, 1]