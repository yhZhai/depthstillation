import torch


def generate_parameters_rotate(num_steps, height=5.0, reverse=False):
    """
    Generate a sequence of axisangle and translation parameters
    for rotating the camera from left to right (or right to left if reversed) while facing the origin.

    Args:
    - num_steps (int): The number of steps in the rotation sequence.
    - height (float): The height (distance) of the camera from the origin.
    - reverse (bool): Whether to reverse the rotation direction.

    Returns:
    - List[str], List[str]: Lists of axisangle and translation strings.
    """
    # Create a vertical axis (Y-axis) for rotation
    axis = torch.tensor([0.0, 1.0, 0.0])

    # Define the angles based on the 'reverse' flag
    if reverse:
        angles = torch.linspace(torch.pi / 2, -torch.pi / 2, steps=num_steps)
    else:
        angles = torch.linspace(-torch.pi / 2, torch.pi / 2, steps=num_steps)

    # For each angle, calculate the axis-angle representation
    axisangles = [axis * angle for angle in angles]

    # Convert axisangle tensors to strings
    axisangle_strings = [",".join(map(str, aa.numpy())) for aa in axisangles]

    # Assuming the camera is always at the same position relative to the origin
    translation = torch.tensor([0.0, 0.0, height])

    # Convert translation tensor to string (constant for all steps)
    translation_string = ",".join(map(str, translation.numpy()))
    translation_strings = [translation_string] * num_steps

    # Replace '-' with 'n' in the strings as in your original version
    axisangle_strings = [aa.replace("-", "n") for aa in axisangle_strings]
    translation_strings = [t.replace("-", "n") for t in translation_strings]

    return axisangle_strings, translation_strings


# Generate parameters for 10 steps
axisangle_sequence, translation_sequence = generate_parameters_rotate(10)
for aa, t in zip(axisangle_sequence, translation_sequence):
    print("AxisAngle:", aa)
    print("Translation:", t)
    print()
