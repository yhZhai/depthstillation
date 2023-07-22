import concurrent.futures
import subprocess
from pathlib import Path

from tqdm import tqdm


def process_image(image_path, depth_path, seg_path, save_path):
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
        "--seg_path",
        str(seg_path),
        "--save_path",
        str(save_path),
        "--padding",
        "75",
        "--binary_segment",
        "--zero_bg_depth",
        "--center_crop_segment",
    ]

    subprocess.run(cmd, check=True)


def process_images_in_folder(
    image_folder: str, depth_folder: str, seg_folder: str, save_path: str, num_workers: int
):
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg"))

    # Determine depth and segmentation file names
    depth_paths = [Path(depth_folder) / image_path.name for image_path in image_paths]
    seg_paths = [
        Path(seg_folder) / image_path.name.replace("png.", ".").replace("jpg.", ".")
        for image_path in image_paths
    ]
    save_path = [Path(save_path)] * len(image_paths)

    # Use a process pool to concurrently process images
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(process_image, image_paths, depth_paths, seg_paths, save_path),
                total=len(image_paths),
            )
        )


if __name__ == "__main__":
    # Adjust the number of workers as needed
    process_images_in_folder(
        # "samples/tiktok/image_gt",
        # "/mnt/c/Users/admin/Documents/meeting_materials/07-01-2023/disco_image_baseline/pred_gs3.0_scale-cond1.0-ref1.0",
        # "/mnt/c/Users/admin/Documents/meeting_materials/07-01-2023/disco_depth_baseline/pred_gs3.0_scale-cond1.0-ref1.0",
        # "samples/tiktok/depth_gt",
        "/mnt/c/Users/admin/Documents/meeting_materials/07-20-2023/clsmidattunet/pred_gs3.0_scale-cond1.0-ref1.0",
        "/mnt/c/Users/admin/Documents/meeting_materials/07-20-2023/clsmidattunet/depth_pred_gs3.0_scale-cond1.0-ref1.0",
        "samples/tiktok/mask",
        "tiktok_clsmidattunet",
        num_workers=12,
    )
