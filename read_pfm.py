import re

import numpy as np


def read_pfm(file_path):
    with open(file_path, "rb") as file:
        # Read the header
        header = file.readline().decode().rstrip()
        color = False

        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        # second line: width height
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        # third line: scale
        scale = float(file.readline().decode().rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # flip the data in the up/down direction
        return data, scale


if __name__ == "__main__":
    depth, scale = read_pfm("samples/tiktok_depth.pfm")
    print("a")
