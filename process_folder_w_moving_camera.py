import concurrent.futures
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

# from generate_camera_positions import generate_camera_positions
# from generate_camera_positions_rotate_camera import generate_parameters_rotate
# from generate_camera_positions_rotate_translate import generate_parameters, orbit_camera_around_center
from generate_camera_positions_rotate_translate import orbit_camera_around_center


def process_image(image_path, depth_path, seg_path, save_path, ab="0,0,0", mb="0,0,0"):
    assert image_path.exists(), f"Image file {image_path} does not exist"
    assert depth_path.exists(), f"Depth file {depth_path} does not exist"
    if not seg_path.exists():
        if seg_path.name.endswith(".jpg"):
            seg_path = Path(str(seg_path).replace(".jpg", ".png"))
        elif seg_path.name.endswith(".png"):
            seg_path = Path(str(seg_path).replace(".png", ".jpg"))
    assert seg_path.exists(), f"Segmentation file {seg_path} does not exist"

    cmd = [
        "python",
        "depthstillation.py",
        "--image_path",
        str(image_path),
        "--depth_path",
        str(depth_path),
        "--save_path",
        str(save_path),
        "--padding",
        "75",
        "--center_crop_segment",
        "-vra",
        "0,0,0",
        "-vrm",
        "0,0,0",
        "-ab",
        ab,
        # "-ab", "0,0,0",
        "-mb",
        mb,
        # "-mb", "0,0,0"
    ]

    subprocess.run(cmd, check=True)


def process_images_in_folder(
    image_folder: str,
    depth_folder: str,
    seg_folder: str,
    save_path: str,
    num_workers: int,
):
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg"))

    # motions, angles = orbit_camera_around_center(20, 0.1)
    # motions = motions * 300
    # motions = motions[: len(image_paths)]
    motions = [0.3 - i * 0.3 / 40 for i in range(80)]
    motions = [f"0,{i},0".replace("-", "n") for i in motions]
    motions = motions + motions[::-1]
    motions = motions * 300
    motions = motions[: len(image_paths)]
    # angles = angles * 300
    # angles = angles[: len(image_paths)]
    angles = [np.pi / 6 - i * np.pi / 6 / 40 for i in range(80)]
    angles = [f"0,{i},0".replace("-", "n") for i in angles]
    angles = angles + angles[::-1]
    angles = angles * 300
    angles = angles[: len(image_paths)]

    print("Motions:", motions[:10])
    print("Angles:", angles[:10])

    # Determine depth and segmentation file names
    depth_paths = []
    for image_path in image_paths:
        if (Path(depth_folder) / image_path.name).exists():
            depth_paths.append(Path(depth_folder) / image_path.name)
        elif (
            Path(depth_folder)
            / image_path.name.replace("png.", ".").replace("jpg.", ".")
        ).exists():
            depth_paths.append(
                Path(depth_folder)
                / image_path.name.replace("png.", ".").replace("jpg.", ".")
            )
        elif (
            Path(depth_folder)
            / image_path.name.replace("png.", ".")
            .replace("jpg.", ".")
            .replace(".png", ".jpg")
        ).exists():
            depth_paths.append(
                Path(depth_folder)
                / image_path.name.replace("png.", ".")
                .replace("jpg.", ".")
                .replace(".png", ".jpg")
            )
        else:
            print(
                Path(depth_folder)
                / image_path.name.replace("png.", ".").replace("jpg.", ".")
            )
            raise FileNotFoundError(f"Depth file for {image_path} not found")
    seg_paths = [
        Path(seg_folder) / image_path.name.replace("png.", ".").replace("jpg.", ".")
        for image_path in image_paths
    ]
    save_path = [Path(save_path)] * len(image_paths)

    # Use a process pool to concurrently process images
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(
                    process_image,
                    image_paths,
                    depth_paths,
                    seg_paths,
                    save_path,
                    angles,
                    motions,
                ),
                total=len(image_paths),
            )
        )


if __name__ == "__main__":
    # Adjust the number of workers as needed
    process_images_in_folder(
        image_folder="samples/tiktok/image_gt",
        depth_folder="samples/tiktok/hdnet_depth_gray_gt",
        seg_folder="samples/tiktok/mask",
        save_path="tiktok_hdnet_gt_moving_camera",
        num_workers=4,
    )
