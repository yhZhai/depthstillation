import subprocess
from pathlib import Path
from tqdm import tqdm


def process_images_in_folder(image_folder: str, depth_folder: str, seg_folder: str):
    # Convert string path to a Path object
    image_folder = Path(image_folder)

    # define your command template
    cmd = [
        "python",
        "depthstillation.py",
        "--image_path",
        None,  # placeholders that we'll fill in the loop
        "--depth_path",
        None,
        "--seg_path",
        None,
        "--save_path",
        "tiktok",
        "--padding",
        "50",
        "--binary_segment",
        # "--save_everything",
        "--zero_bg_depth",
        "--center_crop_segment",
    ]

    # iterate over png files in the folder
    for image_path in tqdm(image_folder.glob("*.png")):
        image_name = image_path.name

        # Construct the paths to the corresponding depth and segmentation files
        depth_path = Path(depth_folder) / image_name
        seg_path = Path(seg_folder) / image_name.replace("png.", ".").replace(
            "jpg.", "."
        )

        assert image_path.exists(), f"Image file {image_path} does not exist"
        assert depth_path.exists(), f"Depth file {depth_path} does not exist"
        assert seg_path.exists(), f"Segmentation file {seg_path} does not exist"

        # replace placeholders with actual paths
        cmd[3] = str(image_path)
        cmd[5] = str(depth_path)
        cmd[7] = str(seg_path)

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    process_images_in_folder(
        "samples/tiktok/image_gt", "samples/tiktok/depth_gt", "samples/tiktok/mask"
    )
