import torch
import numpy as np
import cv2

def lab2rgb(L, AB):
    # Denormalize L channel (which was scaled to [-1, 1] and originally in range [0, 100])
    L = (L + 1) * 50.0  # Now L should be in [0, 100]

    # Denormalize AB channels (which were scaled to [-1, 1] and originally in range [-128, 127])
    AB = AB * 128.0  # Now AB should be in [-128, 127]

    # Convert PyTorch tensors to NumPy arrays and ensure they have the right dtype
    L = L.squeeze().cpu().numpy().astype(np.float32)
    AB = AB.squeeze().cpu().numpy().astype(np.float32)

    # Ensure AB has the correct shape (2, height, width)
    if AB.ndim == 2:
        AB = np.expand_dims(AB, axis=0)

    # Stack L and AB channels to form a LAB image
    lab_image = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    lab_image[:, :, 0] = L
    lab_image[:, :, 1] = AB[0, :, :]
    lab_image[:, :, 2] = AB[1, :, :]

    # Convert LAB image to RGB using OpenCV
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Ensure the RGB image is in the expected range of [0, 255] and dtype
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    # Convert back to PyTorch tensor and normalize to [0, 1] if needed
    rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1)  # Change to C x H x W format for PyTorch

    return rgb_image.float() / 255.0  # Return a float tensor in the range [0, 1]