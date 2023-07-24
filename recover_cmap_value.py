import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def closest_color(rgb, colormap):
    """Find the closest color in the colormap."""
    rgb = np.array(rgb)
    differences = np.sum((colormap - rgb) ** 2, axis=1)
    index_of_smallest_difference = np.argmin(differences)
    return index_of_smallest_difference


def decode_colormap_image(image_path, reverse=False):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    img_data = np.array(img)

    # Generate the 'hot' colormap from matplotlib
    colormap = plt.get_cmap("hot")(np.linspace(0, 1, 256))[
        :, :3
    ]  # Remove alpha if present
    colormap = (colormap * 255).astype(int)

    # For each pixel, get the original value from the closest color in the colormap
    original_values = np.zeros(img_data.shape[:2], dtype=np.float32)
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            original_values[i, j] = closest_color(img_data[i, j], colormap) / 255.0

    # scaled from 0 to 1
    if reverse:
        return 1.0 - original_values
    return original_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode a colormap image back to its original values."
    )
    parser.add_argument("image_path", type=str, help="Path to the colormap image.")
    parser.add_argument("--reverse", action="store_true", help="Reverse the depth.")

    args = parser.parse_args()

    decoded_values = decode_colormap_image(args.image_path, args.reverse)
    print(decoded_values)
