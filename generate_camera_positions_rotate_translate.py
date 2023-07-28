import numpy as np


def compute_rotation_to_face_center(camera_position):
    """
    Given a camera position, compute the axis-angle rotation
    for the camera to face the origin (center of the object).
    """
    # The direction vector from the camera to the origin
    direction = -np.array(camera_position)

    # Normalize the direction
    direction_norm = direction / np.linalg.norm(direction)

    # Calculate the rotation axis: this is the cross product between the
    # direction vector and the up vector (0, 0, 1 in this case)
    rotation_axis = np.cross(direction_norm, [0, 0, 1])
    rotation_axis_norm = rotation_axis / np.linalg.norm(rotation_axis)

    # Calculate the angle of rotation: this is the angle between the
    # direction vector and the up vector
    rotation_angle = np.arccos(np.dot(direction_norm, [0, 0, 1]))

    # Return the axis-angle representation
    return rotation_axis_norm * rotation_angle


def orbit_camera_around_center(total_frames, orbit_radius, constant_z=0):
    """
    Computes the camera's position and rotation for each frame
    to achieve an orbit around the center.
    """
    theta_increment = 2 * np.pi / total_frames
    camera_positions = []
    camera_rotations = []

    for i in range(total_frames):
        theta = i * theta_increment

        # Compute camera translation (position)
        camera_mot = [
            orbit_radius * np.cos(theta),
            orbit_radius * np.sin(theta),
            constant_z,
        ]
        camera_positions.append(",".join(map(str, camera_mot)).replace("-", "n"))

        # Compute camera rotation (axis-angle)
        camera_ang = compute_rotation_to_face_center(camera_mot)
        camera_rotations.append(",".join(map(str, camera_ang)).replace("-", "n"))

    return camera_positions, camera_rotations


# camera_positions, camera_rotations = orbit_camera_around_center(360, 10) # Orbit at a radius of 10 units
# print("Camera positions:", camera_positions[:5])
# print("Camera rotations:", camera_rotations[:5])


# def generate_parameters(num_steps, height=5.0, radius=5.0, reverse=False):
#     """
#     Generate a sequence of axisangle and translation parameters
#     for making the camera orbit around the origin while facing it.
#
#     Args:
#     - num_steps (int): The number of steps in the rotation sequence.
#     - height (float): The height (Y-coordinate) of the camera from the origin.
#     - radius (float): The radius of the circular path around the origin.
#     - reverse (bool): Whether to reverse the rotation direction.
#
#     Returns:
#     - List[str], List[str]: Lists of axisangle and translation strings.
#     """
#     # Create a vertical axis (Y-axis) for rotation
#     axis = torch.tensor([0.0, 1.0, 0.0])
#
#     # Define the angles based on the 'reverse' flag
#     if reverse:
#         angles = torch.linspace(torch.pi / 4, -torch.pi / 4, steps=num_steps)
#     else:
#         angles = torch.linspace(-torch.pi / 4, torch.pi / 4, steps=num_steps)
#
#     # Calculate the axis-angle representation for each angle
#     axisangles = [axis * angle for angle in angles]
#
#     # Convert axisangle tensors to strings
#     axisangle_strings = [",".join(map(str, aa.numpy())) for aa in axisangles]
#
#     # Calculate the camera's position (translation) for each angle using the circle equation
#     translations = [
#         torch.tensor([radius * torch.cos(angle), height, radius * torch.sin(angle)])
#         for angle in angles
#     ]
#
#     # Convert translation tensors to strings
#     translation_strings = [",".join(map(str, t.numpy())) for t in translations]
#
#     # Replace '-' with 'n' in the strings as in your original version
#     axisangle_strings = [aa.replace("-", "n") for aa in axisangle_strings]
#     translation_strings = [t.replace("-", "n") for t in translation_strings]
#
#     return axisangle_strings, translation_strings
#
#
# axisangle_sequence, translation_sequence = generate_parameters(10, 0, 0.2)
# for aa, t in zip(axisangle_sequence, translation_sequence):
#     print("AxisAngle:", aa)
#     print("Translation:", t)
#     print()
#
