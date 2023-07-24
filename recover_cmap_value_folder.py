import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from recover_cmap_value import decode_colormap_image


def process_image(image_file, output_path, reverse):
    decoded_values = decode_colormap_image(image_file, reverse)

    # Convert the float values to uint8 for saving as an image
    decoded_image_data = (decoded_values * 255).astype(np.uint8)

    # Create a PIL image from the numpy array and save
    img = Image.fromarray(
        decoded_image_data, mode="L"
    )  # 'L' mode indicates 8-bit pixels, black and white
    img.save(output_path / image_file.name)


def process_colormap_folder(input_folder, output_folder, reverse=False, num_workers=4):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Using a ProcessPoolExecutor to process images concurrently
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image, image_file, output_path, reverse)
            for image_file in list(input_path.glob("*.png"))
            + list(input_path.glob("*.jpg"))
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of colormap images and save them as grayscale images."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing colormap images."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the folder where grayscale images will be saved.",
    )
    parser.add_argument("--reverse", action="store_true", help="Reverse the depth.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of processes to use. Default is 12.",
    )

    args = parser.parse_args()

    process_colormap_folder(
        args.input_folder, args.output_folder, args.reverse, args.num_workers
    )
