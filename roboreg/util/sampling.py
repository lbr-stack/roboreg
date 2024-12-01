from typing import Tuple

import numpy as np
import torch


def random_position_in_hollow_spheres(
    inner_radii: torch.Tensor,
    outer_radii: torch.Tensor,
) -> torch.Tensor:
    r"""Randomly samples a position in a hollow sphere.

    Args:
        inner_radius (torch.Tensor): The inner radii of the hollow spheres.
        outer_radius (torch.Tensor): The outer radii of the hollow spheres.

    Returns:
        torch.Tensor: A tensor of shape (radii, 3) containing the sampled positions.
    """
    if inner_radii.shape != outer_radii.shape:
        raise ValueError(
            "Expected inner_radius and outer_radius to have the same shape."
        )
    if torch.greater(inner_radii, outer_radii).any():
        raise ValueError(f"Expected inner_radius <= outer_radius.")
    if torch.less(inner_radii, 0.0).any():
        raise ValueError("Expected inner_radius >= 0.")
    r = torch.rand_like(outer_radii) * (outer_radii - inner_radii) + inner_radii
    theta = torch.rand_like(outer_radii) * 2 * torch.full_like(outer_radii, torch.pi)
    phi = torch.rand_like(outer_radii) * torch.full_like(outer_radii, torch.pi)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def random_fov_eye_space_coordinates(
    heights: torch.IntTensor,
    widths: torch.IntTensor,
    focal_lengths_x: torch.Tensor,
    focal_lengths_y: torch.Tensor,
    eye_min_dists: torch.Tensor,
    eye_max_dists: torch.Tensor,
    angle_intervals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Randomly samples eye space coordinates for a camera with a given field of view.
    The sampled eye space coordinates guarantee that the view center is visible under the field of view.

    Args:
        heights (torch.IntTensor): The heights of the images. 1D tensor with N elements.
        widths (torch.IntTensor): The widths of the images. 1D tensor with N elements.
        focal_lengths_x (torch.Tensor): The focal lengths in x direction. 1D tensor with N elements.
        focal_lengths_y (torch.Tensor): The focal lengths in y direction. 1D tensor with N elements.
        eye_min_dists (torch.Tensor): The minimum distances of the eye from the origin. 1D tensor with N elements.
        eye_max_dists (torch.Tensor): The maximum distances of the eye from the origin. 1D tensor with N elements.
        angle_intervals (torch.Tensor): The intervals [-angle_intervals/2, angle_intervals/2] in which to sample the rotation angle. 1D tensor with N elements.

    Returns:
        Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
            - Random eye positions of shape (N, 3).
            - Random center positions of shape (N, 3).
            - Random rotation angles of shape (N, 1).
    """
    if (
        heights.device != widths.device
        or heights.device != focal_lengths_x.device
        or heights.device != focal_lengths_y.device
        or heights.device != eye_min_dists.device
        or heights.device != eye_max_dists.device
        or heights.device != angle_intervals.device
    ):
        raise ValueError("Expected all tensors to be on the same device.")
    if heights.ndim != 1:
        raise ValueError("Expected heights to be a 1D tensor.")
    if (
        heights.shape != widths.shape
        or heights.shape != focal_lengths_x.shape
        or heights.shape != focal_lengths_y.shape
        or heights.shape != eye_min_dists.shape
        or heights.shape != eye_max_dists.shape
        or heights.shape != angle_intervals.shape
    ):
        raise ValueError("Expected all tensors to be of same shape.")

    # compute a random eye position
    random_eye = random_position_in_hollow_spheres(
        inner_radii=eye_min_dists,
        outer_radii=eye_max_dists,
    )

    # compute maximum distance for the view center from the origin
    # such that center is still visible under field of view
    fovs_x = 2 * torch.arctan(widths / (2 * focal_lengths_x))
    fovs_y = 2 * torch.arctan(heights / (2 * focal_lengths_y))
    max_fovs = torch.max(fovs_x, fovs_y)
    distance = torch.norm(random_eye, dim=1)
    center_max_dists = distance * torch.tan(max_fovs / 2) / 2

    # compute a random rotation (parameterized by a center point of view)
    random_center = random_position_in_hollow_spheres(
        inner_radii=torch.zeros_like(center_max_dists),
        outer_radii=center_max_dists,
    )

    random_angle = (
        torch.rand_like(angle_intervals) * angle_intervals - angle_intervals / 2.0
    ).unsqueeze(-1)

    return random_eye, random_center, random_angle
