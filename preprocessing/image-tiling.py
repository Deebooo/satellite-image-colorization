import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from glob import glob

def create_grid(image_path, tile_size=512):
    with rasterio.open(image_path) as src:
        height, width = src.shape
        x_points = range(0, width, tile_size)
        y_points = range(0, height, tile_size)
        grid_windows = []
        for x in x_points:
            for y in y_points:
                window = Window(x, y, min(tile_size, width - x), min(tile_size, height - y))
                grid_windows.append(window)
    return grid_windows

def cut_raster_with_grid(image_paths, output_img_dir, tile_size=512):
    os.makedirs(output_img_dir, exist_ok=True)

    global_counter = 0  # Initialize a global counter for unique naming

    for image_path in tqdm(image_paths, desc="Processing Images"):
        grid_windows = create_grid(image_path, tile_size)

        with rasterio.open(image_path) as src:
            meta = src.meta.copy()

            for window in grid_windows:
                # Read the data in the window
                img_data = src.read(window=window)

                # Update metadata for the tile
                meta.update({
                    "driver": "GTiff",
                    "height": window.height,
                    "width": window.width,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

                output_img_path = os.path.join(output_img_dir, f"img_{global_counter}.tif")

                # Write the tile
                with rasterio.open(output_img_path, 'w', **meta) as dst:
                    dst.write(img_data)

                global_counter += 1  # Increment the global counter after each tile is processed

# Example usage
image_dir = "/local_disk/helios/skhelil/fichiers/images_satt/satt_ori"
output_img_dir = "/local_disk/helios/skhelil/fichiers/images_satt/satt_decoup"
tile_size = 256  # Size in pixels

# Get all image paths from the directory
image_paths = glob(os.path.join(image_dir, "*.tif"))

# Cut the rasters with the grid for all images
cut_raster_with_grid(image_paths, output_img_dir, tile_size)
