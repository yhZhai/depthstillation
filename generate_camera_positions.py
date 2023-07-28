import math

import torch


def generate_camera_positions(radius, steps, height=0):
    """
    Generate separate lists for camera positions and Euler rotation angles for a circular orbit around the origin.

    Parameters:
    - radius: Distance from the origin in the XY plane.
    - height: Height above (or below) the XY plane.
    - steps: Number of positions to generate.

    Returns:
    Two lists:
    1. A list of lists for positions, each sublist containing [x, y, z].
    2. A list of lists for Euler rotation angles in radians, each sublist containing [rx, ry, rz].
    """

    positions = []
    rotations = []

    # Calculate the angle increment for each step over 180 degrees
    angle_increment = math.pi / steps

    for i in range(steps + 1):
        # Calculate the current angle, starting from pi and decreasing
        angle = math.pi - i * angle_increment

        # Convert polar coordinates to Cartesian coordinates
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height

        # Calculate the Euler angles for the camera to face the origin
        ry = angle
        rx = math.atan2(-z, radius)
        rz = 0

        position_str = "{:.5f},{:.5f},{:.5f}".format(x, y, z)
        rotation_str = "{:.5f},{:.5f},{:.5f}".format(rx, ry, rz)
        # rotation_str = modified_euler_to_axisangle(rotation_str)

        position_str = position_str.replace("-", "n")
        rotation_str = rotation_str.replace("-", "n")

        positions.append(position_str)
        rotations.append(rotation_str)

    return positions, rotations


def modified_euler_to_axisangle(rotation_str):
    """Converts Euler angles (in string format) to axis-angle representation (in string format).
    This implementation assumes that rotations are in the order of rx, ry, rz.
    """
    # Convert string representation to tensor
    rotations = torch.tensor(list(map(float, rotation_str.split(","))))

    # Convert rotations to rotation matrices
    cos = torch.cos(rotations)
    sin = torch.sin(rotations)

    # Rotation matrix around X-axis
    Rx = torch.tensor([[1, 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]])

    # Rotation matrix around Y-axis
    Ry = torch.tensor([[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]])

    # Rotation matrix around Z-axis
    Rz = torch.tensor([[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, 1]])

    # Combined rotation matrix
    R = torch.matmul(Rz, torch.matmul(Ry, Rx))

    # Convert rotation matrix to axis-angle
    angle = torch.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)

    # Handle the special case of no rotation
    if angle.item() == 0:
        axisangle = torch.tensor([1.0, 0.0, 0.0]) * angle
    else:
        x = (R[2, 1] - R[1, 2]) / (2 * torch.sin(angle))
        y = (R[0, 2] - R[2, 0]) / (2 * torch.sin(angle))
        z = (R[1, 0] - R[0, 1]) / (2 * torch.sin(angle))
        axisangle = torch.tensor([x, y, z]) * angle

    # Convert tensor result to string format
    axisangle_str = ",".join(map(str, axisangle.tolist()))
    return axisangle_str


if __name__ == "__main__":
    # Example usage
    radius = 10
    height = 0
    steps = 12
    positions, rotations = generate_camera_positions(radius, steps, height)

    print("Positions:")
    for pos in positions:
        print(pos)

    print("\nEuler Rotations (in radians):")
    for rot in rotations:
        print(rot)
