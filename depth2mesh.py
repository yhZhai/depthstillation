import argparse
import os

import cv2
import numpy as np
import pyvista as pv
import trimesh

from recover_cmap_value import decode_colormap_image


def depth_map_to_mesh(
    depth_map_path,
    texture_image_path,
    output_obj_path,
    scale_factor=1.0,
    ply=False,
    center_crop=False,
):
    # Load the depth map
    depth_map = cv2.imread(str(depth_map_path), -1)
    if len(depth_map.shape) == 3:  # rgb image
        depth_map = np.mean(depth_map, axis=2)

    if depth_map.shape[0] != depth_map.shape[1] and center_crop:
        min_dim = min(depth_map.shape[0], depth_map.shape[1])
        depth_map = depth_map[
            (depth_map.shape[0] - min_dim) // 2 : (depth_map.shape[0] + min_dim) // 2,
            (depth_map.shape[1] - min_dim) // 2 : (depth_map.shape[1] + min_dim) // 2,
        ]

    h, w = depth_map.shape

    with open(output_obj_path, "w") as obj_file:
        # Write vertices to .obj file
        for i in range(h):
            for j in range(w):
                # Convert depth (pixel intensity) to Z coordinate
                z = depth_map[i, j] * scale_factor
                obj_file.write(f"v {j} {-i} {z}\n")

        # Write UV coordinates
        for i in range(h):
            for j in range(w):
                # Texture coordinates (scaled between 0 and 1)
                u = j / (w - 1)
                v = 1 - (i / (h - 1))
                obj_file.write(f"vt {u} {v}\n")

        # If texture_image_path is provided, save it
        if texture_image_path:
            texture_image = cv2.imread(str(texture_image_path))
            assert (
                texture_image.shape[:2] == depth_map.shape
            ), "Texture image size doesn't match the depth map!"
            output_texture_path = output_obj_path.replace(".obj", ".jpg")
            cv2.imwrite(output_texture_path, texture_image)
            mtl_path = output_obj_path.replace(".obj", ".mtl")
            with open(mtl_path, "w") as mtl_file:
                mtl_file.write(f"newmtl TextureMaterial\n")
                mtl_file.write(f"Ka 1.000 1.000 1.000\n")  # Ambient color
                mtl_file.write(f"Kd 1.000 1.000 1.000\n")  # Diffuse color
                mtl_file.write(f"Ks 0.000 0.000 0.000\n")  # Specular color
                mtl_file.write(f"illum 1\n")
                mtl_file.write(f"map_Kd {output_texture_path}\n")  # Texture path
            with open(output_obj_path, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f"mtllib {mtl_path}\n")
                f.write(f"usemtl TextureMaterial\n")
                f.write(content)

        # Write faces to .obj file (always with texture coordinates)
        for i in range(h - 1):
            for j in range(w - 1):
                idx1 = i * w + j + 1
                idx2 = (i + 1) * w + j + 1
                idx3 = (i + 1) * w + j + 2
                idx4 = i * w + j + 2
                obj_file.write(f"f {idx1}/{idx1} {idx2}/{idx2} {idx3}/{idx3}\n")
                obj_file.write(f"f {idx1}/{idx1} {idx3}/{idx3} {idx4}/{idx4}\n")

        # Save as .ply if the ply flag is set
        if ply:
            output_mesh_path = output_obj_path.replace(".obj", ".ply")
            # print(f"Saving mesh to {output_mesh_path}")
            convert_obj_to_ply(output_obj_path, output_mesh_path)
            # Delete original .obj and .mtl files
            if os.path.exists(output_obj_path):
                os.remove(output_obj_path)
            mtl_path = output_obj_path.replace(".obj", ".mtl")
            if os.path.exists(mtl_path):
                os.remove(mtl_path)


def convert_obj_to_ply(obj_path, ply_path):
    # Load the mesh from the .obj file
    mesh = trimesh.load(obj_path, force="mesh")

    # Check if the mesh has texture coordinates
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        # If it does, make sure to preserve them when converting
        # This step can be expanded upon if there are other specifics you want to address
        pass

    # Save as .ply
    mesh.export(ply_path, file_type="ply")


# def convert_obj_to_ply(obj_path, ply_path):
#     mesh = pv.read(obj_path)
#
#     # If the mesh has texture coordinates, map them to the .ply
#     if hasattr(mesh, "t_coords"):
#         mesh.point_data["Texture Coordinates"] = mesh.t_coords
#
#     mesh.save(ply_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a depth map to a 3D mesh in .OBJ format with optional texture."
    )
    parser.add_argument("depth_map_path", type=str, help="Path to the depth map image.")
    parser.add_argument(
        "-t",
        "--texture_image_path",
        default=None,
        type=str,
        help="Path to the texture image (optional).",
    )
    parser.add_argument(
        "-o",
        "--output_obj_path",
        default="",
        type=str,
        help="Path to save the resulting .OBJ file.",
    )
    parser.add_argument(
        "-s",
        "--scale_factor",
        type=float,
        default=1.0,
        help="Factor to scale the depth values. Default is 1.0.",
    )
    parser.add_argument(
        "--ply",
        action="store_true",
        help="Flag indicating whether to save the resulting mesh in .PLY format instead of .OBJ.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Center crop depth map",
    )

    args = parser.parse_args()
    if args.output_obj_path == "":
        args.output_obj_path = args.depth_map_path.replace(".png", ".obj").replace(
            ".jpg", ".obj"
        )

    depth_map_to_mesh(
        args.depth_map_path,
        args.texture_image_path,
        args.output_obj_path,
        args.scale_factor,
        args.ply,
        args.center_crop,
    )
