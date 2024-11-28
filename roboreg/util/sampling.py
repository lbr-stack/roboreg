from typing import Tuple

import numpy as np
import torch


def random_position_in_hollow_sphere(
    inner_radius: float,
    outer_radius: float,
    batch_size: int = 1,
    device: torch.device = "cuda",
) -> torch.Tensor:
    r"""Randomly samples a position in a hollow sphere.

    Args:
        inner_radius (float): The inner radius of the hollow sphere.
        outer_radius (float): The outer radius of the hollow sphere.
        batch_size (int): The number of samples to generate. Defaults to 1.
        device (torch.device): The device to generate the samples on. Defaults to "cuda".

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3) containing the sampled positions.
    """
    if inner_radius > outer_radius:
        raise ValueError(
            f"Expected inner_radius <= outer_radius, got inner_radius={inner_radius} and outer_radius={outer_radius}."
        )
    if inner_radius < 0:
        raise ValueError("Expected inner_radius >= 0.")
    if outer_radius <= 0:
        raise ValueError("Expected outer_radius > 0.")
    r = (
        torch.rand(batch_size, device=device) * (outer_radius - inner_radius)
        + inner_radius
    )
    theta = (
        torch.rand(batch_size, device=device)
        * 2
        * torch.tensor([torch.pi], device=device)
    )
    phi = torch.rand(batch_size, device=device) * torch.tensor(
        [torch.pi], device=device
    )
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def random_fov_eye_space_coordinates(
    height: int,
    width: int,
    focal_length_x: float,
    focal_length_y: float,
    eye_min_dist: float = 1.0,
    eye_max_dist: float = 5.0,
    angle_interval: float = torch.pi,
    batch_size: int = 1,
    device: torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Randomly samples eye space coordinates for a camera with a given field of view.
    The sampled eye space coordinates guarantee that the view center is visible under the field of view.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        focal_length_x (float): The focal length in x direction.
        focal_length_y (float): The focal length in y direction.
        eye_min_dist (float): The minimum distance of the eye from the origin. Defaults to 1.0.
        eye_max_dist (float): The maximum distance of the eye from the origin. Defaults to 5.0.
        angle_interval (float): The interval [-angle_interval/2, angle_interval/2] in which to sample the rotation angle. Defaults to pi.
        batch_size (int): The number of samples to generate. Defaults to 1.
        device (torch.device): The device to generate the samples on. Defaults to "cuda".

    Returns:
        Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
            - Random eye positions of shape (batch_size, 3).
            - Random center positions of shape (batch_size, 3).
            - Random rotation angles of shape (batch_size, 1).
    """
    # compute a random eye position
    random_eye = random_position_in_hollow_sphere(
        inner_radius=eye_min_dist,
        outer_radius=eye_max_dist,
        batch_size=batch_size,
        device=device,
    )

    # compute maximum distance for the view center from the origin
    # such that center is still visible under field of view
    fov_x = 2 * np.arctan(width / (2 * focal_length_x))
    fov_y = 2 * np.arctan(height / (2 * focal_length_y))
    max_fov = max(fov_x, fov_y)
    distance = torch.norm(random_eye, dim=1)
    center_max_dist = distance * np.tan(max_fov / 2) / 2

    # compute a random rotation (parameterized by a center point of view)
    random_center = random_position_in_hollow_sphere(
        inner_radius=0.0,
        outer_radius=center_max_dist,
        batch_size=batch_size,
        device=device,
    )

    random_angle = (
        torch.rand(batch_size, device=device) * angle_interval - angle_interval / 2.0
    ).unsqueeze(-1)

    return random_eye, random_center, random_angle
