import argparse
import os
from pathlib import Path

import cv2


def images_to_video(input_folder, output_path, fps=30):
    """
    Convert images in the specified folder to a video.

    Args:
    - input_folder (str): Path to the folder containing frame images.
    - output_path (str): Path to save the output video.
    - fps (int): Frames per second for the output video.
    """

    # Sort the image files
    images = sorted(Path(input_folder).glob("*"))

    # Read the first image to get the width and height
    frame = cv2.imread(str(images[0]))
    h, w, layers = frame.shape
    size = (w, h)

    if output_path.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    else:
        fourcc = cv2.VideoWriter_fourcc(
            *"DIVX"
        )  # Original codec, you can change this as per need

    # Create a video writer object
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for image_path in images:
        img = cv2.imread(str(image_path))
        out.write(img)

    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into a video."
    )

    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing frame images."
    )
    parser.add_argument("output_path", type=str, help="Path to save the output video.")
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the video. Default is 30.",
    )

    args = parser.parse_args()

    images_to_video(args.input_folder, args.output_path, args.fps)
