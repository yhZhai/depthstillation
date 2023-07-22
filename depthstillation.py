# Ctypes package used to call the forward warping C library
import argparse
import ctypes
import math
import os
import random
from ctypes import *
from pathlib import Path

import cv2

# Some packages we use
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from numpy.ctypeslib import ndpointer

# External scripts
from bilateral_filter import sparse_bilateral_filtering
from flow_colors import *
from geometry import *
from read_pfm import read_pfm
from utils import *

# Import warping library
lib = cdll.LoadLibrary("external/forward_warping/libwarping.so")
warp = lib.forward_warping


def parser_argument():
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Depthstillation options")

    parser.add_argument(
        "--image_path", type=str, default="samples/im0.jpg", help="Image path"
    )
    parser.add_argument(
        "--depth_path", type=str, default="samples/d0.png", help="Depth path"
    )
    parser.add_argument(
        "--seg_path", type=str, default="samples/s0.png", help="Segmentation path"
    )
    parser.add_argument("--save_path", type=str, default="dCOCO", help="Save path")
    parser.add_argument("--save_name", type=str, default="", help="Save name")

    parser.add_argument("--padding", type=int, default=50, help="Padding")
    parser.add_argument(
        "--num_motions",
        dest="num_motions",
        type=int,
        help="Number of motions",
        default=1,
    )
    parser.add_argument(
        "--segment",
        dest="segment",
        action="store_true",
        help="Enable segmentation (for moving objects)",
    )
    parser.add_argument(
        "--binary_segment",
        action="store_true",
        help="Binary segmentation to separate background and foreground",
    )
    parser.add_argument(
        "--center_crop_segment",
        action="store_true",
        help="Center crop segmentation to square",
    )
    parser.add_argument(
        "--zero_bg_depth", action="store_true", help="Zero depth on background"
    )
    parser.add_argument(
        "--mask_type",
        dest="mask_type",
        type=str,
        default="H'",
        help="Select mask type",
        choices=["H", "H'"],
    )
    parser.add_argument(
        "--num_objects",
        dest="num_objects",
        type=int,
        help="Number of moving objects",
        default=1,
    )
    parser.add_argument(
        "--no_depth",
        dest="no_depth",
        action="store_true",
        help="Assumes constant depth",
    )
    parser.add_argument(
        "--no_sharp",
        dest="no_sharp",
        action="store_true",
        help="Disable depth sharpening",
    )
    parser.add_argument(
        "--change_k",
        dest="change_k",
        action="store_true",
        help="Use a different K matrix",
    )
    parser.add_argument(
        "--change_motion",
        dest="change_motion",
        action="store_true",
        help="Sample a different random motion",
    )
    parser.add_argument(
        "--save_everything",
        action="store_true",
        default=False,
        help="Save all intermediate images",
    )
    parser.add_argument("--seed", type=int, help="Random seed", default=1024)
    args = parser.parse_args()

    # if num_motions greater than 1, ignore change_motion setting
    if args.num_motions > 1:
        args.change_motion = False

    if args.save_name == "":
        args.save_name = Path(args.image_path).stem

    if args.binary_segment:
        args.segment = True

    return args


def create_directories(args):
    if args.save_everything:
        dir_list = [
            "im0",
            "im1_raw",
            "im1",
            "flow",
            "flow_color",
            "depth_color",
            "instances_color",
            "H",
            "M",
            "M'",
            "P",
            "H'",
        ]
    else:
        dir_list = [
            "im0",
            "im1_raw",
            "im1",
        ]
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for dir_name in dir_list:
        if not os.path.exists(os.path.join(args.save_path, dir_name)):
            os.makedirs(os.path.join(args.save_path, dir_name))


def add_padding(image, pad_size):
    """
    Add zero padding around an image.
    
    Parameters:
    - image: The input image as a NumPy array.
    - pad_size: The padding size as an integer.
    
    Returns:
    - Padded image as a NumPy array.
    """
    
    # Check if the image is grayscale or color
    if len(image.shape) == 2:  # Grayscale
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    else:  # Color image
        pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    return padded_image


def set_random_seeds(args):
    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def open_image(args):
    image_path = args.image_path
    # Open I0 image
    rgb = cv2.imread(image_path, -1)
    if len(rgb.shape) < 3:
        h, w = rgb.shape
        rgb = np.stack((rgb, rgb, rgb), -1)
    else:
        h, w, _ = rgb.shape

    rgb = add_padding(rgb, args.padding)
    return rgb, h, w


def open_depth(args, rgb, h, w):
    depth_path = args.depth_path
    # Open D0 (inverse) depth map and resize to I0
    if depth_path.endswith(".pfm"):
        depth, _ = read_pfm(depth_path)
        depth = normalize_array(depth, 0, 2**16 - 1)
        depth = depth / (2**16 - 1)
    else:
        depth = cv2.imread(depth_path, -1)
        if len(depth.shape) == 2:
            depth = depth / (2**16 - 1)  # read the image as is
        else:  # rgb image
            depth = np.mean(depth, axis=2) / (2**8 - 1)

    if depth.shape[0] != h or depth.shape[1] != w:
        depth = cv2.resize(depth, (w, h))

    depth = add_padding(depth, args.padding)

    # Get depth map and normalize
    depth = 1.0 / (depth + 0.005)
    depth[depth > 100] = 100

    # Set depth to constant value in case we do not want to use depth
    if args.no_depth:
        depth = depth * 0.0 + 1.0

    # Depth sharpening (bilateral filter)
    if not args.no_sharp:
        depth = sparse_bilateral_filtering(
            depth.copy(),
            rgb.copy(),
            filter_size=[5, 5],
            num_iter=2,
        )
    return depth


def get_seg_mask(args, h, w, depth):
    seg_path = args.seg_path
    labels = None
    instances = None
    instances_mask = None
    # Load segmentation mask in case we simulate moving objects
    if args.segment:
        labels = []
        # instances_mask = cv2.imread(seg_path, -1)
        instances_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        if (
            instances_mask.shape[0] != instances_mask.shape[1]
        ) and args.center_crop_segment:
            # Center crop segmentation mask to square
            min_dim = min(instances_mask.shape[0], instances_mask.shape[1])
            instances_mask = instances_mask[
                (instances_mask.shape[0] - min_dim)
                // 2 : (instances_mask.shape[0] + min_dim)
                // 2,
                (instances_mask.shape[1] - min_dim)
                // 2 : (instances_mask.shape[1] + min_dim)
                // 2,
            ]

        # Resize instance mask to I0
        if instances_mask.shape[0] != h or instances_mask.shape[1] != w:
            instances_mask = cv2.resize(instances_mask, (w, h))
        
        instances_mask = add_padding(instances_mask, args.padding)

        if args.binary_segment:
            # Convert to binary mask
            instances_mask = (instances_mask > 0).astype(np.uint8)
        
        if args.zero_bg_depth:
            # Set depth to constant value in case we do not want to use depth
            depth[np.where(instances_mask == 0)] = 100

        # Get total number of objects
        classes = instances_mask.max()

        # Get pixels count for each object
        areas = np.array(
            [instances_mask[instances_mask == c].sum() for c in range(classes + 1)],
            np.float32,
        )

        # If we have any object
        if areas.shape[0] > 1:
            # Keep args.num_objects labels having the largest amount of pixels
            labels = areas.argsort()[-args.num_objects :][::-1]
            instances = []

            # For each object kept
            for l in labels:
                # Create a segmentation mask for the single object
                seg_mask = np.zeros_like(instances_mask)

                # Set to 1 pixels having label l
                seg_mask[instances_mask == l] = 1
                seg_mask = np.expand_dims(seg_mask, 0)

                # Cast to pytorch tensor and append to masks list
                seg_mask = torch.from_numpy(np.stack((seg_mask, seg_mask), -1)).float()
                instances.append(seg_mask)
    return labels, instances, instances_mask, depth


def get_intrinsics(args, h, w):
    # Fix a plausible K matrix
    K = np.array(
        [[[0.58, 0, 0.5, 0], [0, 0.58, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        dtype=np.float32,
    )

    # Fix a different K matrix in case
    if args.change_k:
        K = np.array(
            [[[1.16, 0, 0.5, 0], [0, 1.16, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
            dtype=np.float32,
        )
    K[:, 0, :] *= w
    K[:, 1, :] *= h
    inv_K = torch.from_numpy(np.linalg.pinv(K))
    K = torch.from_numpy(K)

    return K, inv_K


def loop_over_motions(
    args, rgb, h, w, depth, inv_K, K, labels, instances, instances_mask
):
    # Cast I0 and D0 to pytorch tensors
    rgb = torch.from_numpy(np.expand_dims(rgb, 0))
    depth = torch.from_numpy(np.expand_dims(depth, 0)).float()

    # Create objects in charge of 3D projection
    backproject_depth = BackprojectDepth(1, h, w)
    project_3d = Project3D(1, h, w)

    # Prepare p0 coordinates
    meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
    p0 = np.stack(meshgrid, axis=-1).astype(np.float32)

    # Loop over the number of motions
    for idm in range(args.num_motions):
        # Initiate masks dictionary
        masks = {}

        # Sample random motion (twice, if you want a new one)
        sample_motions = 2 if args.change_motion else 1
        for mot in range(sample_motions):
            # Generate random vector t
            # Random sign
            scx = (-1) ** random.randrange(2)
            scy = (-1) ** random.randrange(2)
            scz = (-1) ** random.randrange(2)
            # Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
            cx = (random.random() * 0.1) * scx
            cy = (random.random() * 0.1) * scy
            cz = (random.random() * 0.1) * scz
            camera_mot = [0, cy, 0]

            # generate random triplet of Euler angles
            # Random sign
            sax = (-1) ** random.randrange(2)
            say = (-1) ** random.randrange(2)
            saz = (-1) ** random.randrange(2)
            # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
            ax = (random.random() * math.pi / 36.0 + math.pi / 36.0) * sax
            ay = (random.random() * math.pi / 36.0 + math.pi / 36.0) * say
            az = (random.random() * math.pi / 36.0 + math.pi / 36.0) * saz
            camera_ang = [0, ay, 0]

        axisangle = torch.from_numpy(np.array([[camera_ang]], dtype=np.float32))
        translation = torch.from_numpy(np.array([[camera_mot]]))

        # Compute (R|t)
        T1 = transformation_from_parameters(axisangle, translation)

        # Back-projection
        cam_points = backproject_depth(depth, inv_K)

        # Apply transformation T_{0->1}
        p1, z1 = project_3d(cam_points, K, T1)
        z1 = z1.reshape(1, h, w)

        # Simulate objects moving independently
        if args.segment:
            # Loop over objects
            for l in range(len(labels)):
                sign = 1
                # We multiply the sign by -1 to obtain a motion similar to the one shown in the supplementary (not exactly the same). Can be removed for general-purpose use
                if not args.no_depth:
                    sign = -1

                # Random t (scalars and signs). Zeros and small motions are avoided as before
                cix = (random.random() * 0.05 + 0.05) * (
                    sign * (-1) ** random.randrange(2)
                )
                ciy = (random.random() * 0.05 + 0.05) * (
                    sign * (-1) ** random.randrange(2)
                )
                ciz = (random.random() * 0.05 + 0.05) * (
                    sign * (-1) ** random.randrange(2)
                )
                camerai_mot = [cix, ciy, 0]

                # Random Euler angles (scalars and signs). Zeros and small rotations are avoided as before
                aix = (random.random() * math.pi / 72.0 + math.pi / 72.0) * (
                    sign * (-1) ** random.randrange(2)
                )
                aiy = (random.random() * math.pi / 72.0 + math.pi / 72.0) * (
                    sign * (-1) ** random.randrange(2)
                )
                aiz = (random.random() * math.pi / 72.0 + math.pi / 72.0) * (
                    sign * (-1) ** random.randrange(2)
                )
                camerai_ang = [aix, aiy, 0]

                ai = torch.from_numpy(np.array([[camerai_ang]], dtype=np.float32))
                tri = torch.from_numpy(np.array([[camerai_mot]]))

                # Compute (R|t)
                Ti = transformation_from_parameters(axisangle + ai, translation + tri)

                # Apply transformation T_{0->\pi_i}
                pi, zi = project_3d(cam_points, K, Ti)

                # If a pixel belongs to object label l, replace coordinates in I1...
                p1[instances[l] > 0] = pi[instances[l] > 0]

                # ... and its depth
                zi = zi.reshape(1, h, w)
                z1[instances[l][:, :, :, 0] > 0] = zi[instances[l][:, :, :, 0] > 0]

        # Bring p1 coordinates in [0,W-1]x[0,H-1] format
        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= w - 1
        p1[:, :, :, 1] *= h - 1

        # Create auxiliary data for warping
        dlut = torch.ones(1, h, w).float() * 1000
        safe_y = np.maximum(np.minimum(p1[:, :, :, 1].long(), h - 1), 0)
        safe_x = np.maximum(np.minimum(p1[:, :, :, 0].long(), w - 1), 0)
        warped_arr = np.zeros(h * w * 5).astype(np.uint8)
        img = rgb.reshape(-1)

        # Call forward warping routine (C code)
        warp(
            c_void_p(img.numpy().ctypes.data),
            c_void_p(safe_x[0].numpy().ctypes.data),
            c_void_p(safe_y[0].numpy().ctypes.data),
            c_void_p(z1.reshape(-1).numpy().ctypes.data),
            c_void_p(warped_arr.ctypes.data),
            c_int(h),
            c_int(w),
        )
        warped_arr = warped_arr.reshape(1, h, w, 5).astype(np.uint8)

        # Warped image
        im1_raw = warped_arr[0, :, :, 0:3]

        # Validity mask H
        masks["H"] = warped_arr[0, :, :, 3:4]

        # Collision mask M
        masks["M"] = warped_arr[0, :, :, 4:5]
        # Keep all pixels that are invalid (H) or collide (M)
        masks["M"] = 1 - (masks["M"] == masks["H"]).astype(np.uint8)

        # Dilated collision mask M'
        kernel = np.ones((3, 3), np.uint8)
        masks["M'"] = cv2.dilate(masks["M"], kernel, iterations=1)
        masks["P"] = (np.expand_dims(masks["M'"], -1) == masks["M"]).astype(np.uint8)

        # Final mask P
        masks["H'"] = masks["H"] * masks["P"]

        im1 = cv2.inpaint(im1_raw, 1 - masks[args.mask_type], 3, cv2.INPAINT_TELEA)

        cv2.imwrite(
            os.path.join(args.save_path, "im0", f"{args.save_name}.jpg"), rgb[0].numpy()
        )
        cv2.imwrite(
            os.path.join(args.save_path, "im1_raw", f"{args.save_name}_{idm:02d}.jpg"),
            im1_raw,
        )
        cv2.imwrite(
            os.path.join(args.save_path, "im1", f"{args.save_name}_{idm:02d}.jpg"), im1
        )

        if args.save_everything:
            # Compute flow as p1-p0
            flow_01 = p1 - p0

            # Get 16-bit flow (KITTI format) and colored flows
            flow_16bit = cv2.cvtColor(
                np.concatenate(
                    (flow_01 * 64.0 + (2**15), np.ones_like(flow_01)[:, :, :, 0:1]),
                    -1,
                )[0],
                cv2.COLOR_BGR2RGB,
            )
            flow_color = flow_to_color(flow_01[0].numpy(), convert_to_bgr=True)

            # Save images
            cv2.imwrite(
                os.path.join(args.save_path, "flow", f"{args.save_name}_{idm:02d}.png"),
                flow_16bit.astype(np.uint16),
            )
            cv2.imwrite(
                os.path.join(args.save_path, "H", f"{args.save_name}_{idm:02d}.png"),
                masks["H"] * 255,
            )
            cv2.imwrite(
                os.path.join(args.save_path, "M", f"{args.save_name}_{idm:02d}.png"),
                masks["M"] * 255,
            )
            cv2.imwrite(
                os.path.join(args.save_path, "M'", f"{args.save_name}_{idm:02d}.png"),
                masks["M'"] * 255,
            )
            cv2.imwrite(
                os.path.join(args.save_path, "P", f"{args.save_name}_{idm:02d}.png"),
                masks["P"] * 255,
            )
            cv2.imwrite(
                os.path.join(args.save_path, "H'", f"{args.save_name}_{idm:02d}.png"),
                masks["H'"] * 255,
            )
            cv2.imwrite(
                os.path.join(
                    args.save_path, "flow_color", f"{args.save_name}_{idm:02d}.png"
                ),
                flow_color,
            )
            plt.imsave(
                os.path.join(
                    args.save_path, "depth_color", f"{args.save_name}_{idm:02d}.png"
                ),
                1.0 / depth[0].detach().numpy(),
                cmap="magma",
            )
            if args.segment:
                plt.imsave(
                    os.path.join(
                        args.save_path,
                        "instances_color",
                        f"{args.save_name}_{idm:02d}.png",
                    ),
                    instances_mask,
                    cmap="magma",
                )

        # Clear cache and update progress bar
        ctypes._reset_cache()


def main():
    args = parser_argument()
    create_directories(args)
    set_random_seeds(args)
    rgb, h, w = open_image(args)
    depth = open_depth(args, rgb, h, w)
    labels, instances, instances_mask, depth = get_seg_mask(args, h, w, depth)
    h, w = h + 2 * args.padding, w + 2 * args.padding
    K, inv_K = get_intrinsics(args, h, w)
    loop_over_motions(
        args, rgb, h, w, depth, inv_K, K, labels, instances, instances_mask
    )


if __name__ == "__main__":
    main()
