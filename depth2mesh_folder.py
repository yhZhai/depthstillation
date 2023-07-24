import argparse
import concurrent.futures
from pathlib import Path

from tqdm import tqdm

from depth2mesh import depth_map_to_mesh


def process_single_depth_map(depth_map_file, output_folder, scale_factor):
    output_obj_file = output_folder / depth_map_file.with_suffix(".obj").name
    depth_map_to_mesh(depth_map_file, output_obj_file, scale_factor)


def process_depth_maps_in_folder(
    input_folder, output_folder, scale_factor=1.0, workers=None
):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    depth_map_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Use tqdm to show progress, and `executor.map` to process images in parallel
        list(
            tqdm(
                executor.map(
                    process_single_depth_map,
                    depth_map_files,
                    [output_path] * len(depth_map_files),
                    [scale_factor] * len(depth_map_files),
                ),
                total=len(depth_map_files),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of depth maps and convert them to 3D meshes in .OBJ format using multiprocessing."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing depth map images."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the folder where .OBJ files will be saved.",
    )
    parser.add_argument(
        "-s",
        "--scale_factor",
        type=float,
        default=1.0,
        help="Factor to scale the depth values. Default is 1.0.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=12,
        help="Number of worker processes. If not set, it will use as many cores as available on the machine.",
    )

    args = parser.parse_args()

    process_depth_maps_in_folder(
        args.input_folder, args.output_folder, args.scale_factor, args.workers
    )
