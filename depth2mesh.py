import argparse

import cv2
import numpy as np


def depth_map_to_mesh(depth_map_path, output_obj_path, scale_factor=1.0):
    # Load the depth map
    depth_map = cv2.imread(depth_map_path, -1)
    if len(depth_map.shape) == 2:
        depth_map = depth_map
    else:  # rgb image
        depth_map = np.mean(depth_map, axis=2)

    # Get the shape of the depth map
    h, w = depth_map.shape

    # Open the output .obj file for writing
    with open(output_obj_path, "w") as obj_file:
        # Write vertices to .obj file
        for i in range(h):
            for j in range(w):
                # Convert depth (pixel intensity) to Z coordinate
                z = depth_map[i, j] * scale_factor
                obj_file.write(f"v {j} {-i} {z}\n")

        # Write faces to .obj file
        for i in range(h - 1):
            for j in range(w - 1):
                idx1 = i * w + j + 1
                idx2 = (i + 1) * w + j + 1
                idx3 = (i + 1) * w + j + 2
                idx4 = i * w + j + 2
                obj_file.write(f"f {idx1} {idx2} {idx3}\n")
                obj_file.write(f"f {idx1} {idx3} {idx4}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a depth map to a 3D mesh in .OBJ format."
    )
    parser.add_argument("depth_map_path", type=str, help="Path to the depth map image.")
    parser.add_argument(
        "output_obj_path", type=str, help="Path to save the resulting .OBJ file."
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="Factor to scale the depth values. Default is 1.0.",
    )

    args = parser.parse_args()

    depth_map_to_mesh(args.depth_map_path, args.output_obj_path, args.scale_factor)
