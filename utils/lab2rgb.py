import torch
import numpy as np
import cv2

def lab2rgb(L, AB):
    # Denormalize L channel
    L = (L + 1) * 50.0

    # Denormalize AB channels
    AB = AB * 128.0

    # Convert PyTorch tensors to NumPy arrays
    L = L.cpu().numpy().astype(np.float32)
    AB = AB.cpu().numpy().astype(np.float32)

    # Initialize an empty array for the LAB image
    batch_size = L.shape[0]
    height = L.shape[2]
    width = L.shape[3]

    lab_image = np.zeros((batch_size, height, width, 3), dtype=np.float32)

    # Assign L to the first channel of the LAB image
    lab_image[:, :, :, 0] = L.squeeze(1)

    # Assign AB to the second and third channels of the LAB image
    lab_image[:, :, :, 1:] = np.transpose(AB, (0, 2, 3, 1))

    # Convert LAB image to RGB using OpenCV
    rgb_images = []
    for i in range(batch_size):
        rgb_image = cv2.cvtColor(lab_image[i], cv2.COLOR_LAB2RGB)
        rgb_images.append(rgb_image)

    # Stack the list of RGB images into a single NumPy array and convert to torch tensor
    rgb_images = np.stack(rgb_images, axis=0)
    rgb_images = torch.from_numpy(rgb_images).permute(0, 3, 1, 2)

    return rgb_images.float()  # Return a float tensor in the range [0, 1]