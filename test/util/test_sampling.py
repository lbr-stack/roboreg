import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import torch

from roboreg.util import (
    random_fov_eye_space_coordinates,
    random_position_in_hollow_spheres,
)


def test_random_position_in_hollow_spheres() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inner_radii = torch.tensor([-1.0], device=device)
    outer_radii = torch.tensor([1.0], device=device)

    try:
        random_position_in_hollow_spheres(
            inner_radii=inner_radii, outer_radii=outer_radii
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")

    inner_radii = torch.tensor([1.0], device=device)
    outer_radii = inner_radii / 2.0

    try:
        random_position_in_hollow_spheres(
            inner_radii=inner_radii, outer_radii=outer_radii
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")

    n_radii = 2
    inner_radii = torch.rand(n_radii, device=device)
    outer_radii = inner_radii + 1.0

    position = random_position_in_hollow_spheres(
        inner_radii=inner_radii,
        outer_radii=outer_radii,
    )

    if position.shape[0] != n_radii:
        raise ValueError(f"Expected batch size {n_radii}, got {position.shape[0]}")


def test_random_fov_eye_space_coordinates() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    heights = torch.full([batch_size], 1, device=device)
    widths = torch.full([batch_size], 1, device=device)
    focal_lengths_x = torch.full([batch_size], 1.0, device=device)
    focal_lengths_y = torch.full([batch_size], 1.0, device=device)
    eye_min_dists = torch.full([batch_size], 1.0, device=device)
    eye_max_dists = torch.full([batch_size], 5.0, device=device)
    angle_intervals = torch.full([batch_size], torch.pi, device=device)

    random_eyes, random_centers, random_angles = random_fov_eye_space_coordinates(
        heights=heights,
        widths=widths,
        focal_lengths_x=focal_lengths_x,
        focal_lengths_y=focal_lengths_y,
        eye_min_dists=eye_min_dists,
        eye_max_dists=eye_max_dists,
        angle_intervals=angle_intervals,
    )

    if random_eyes.shape != (batch_size, 3):
        raise ValueError(f"Expected shape {(batch_size, 3)}, got {random_eyes.shape}")
    if random_centers.shape != (batch_size, 3):
        raise ValueError(
            f"Expected shape {(batch_size, 3)}, got {random_centers.shape}"
        )
    if random_angles.shape != (batch_size, 1):
        raise ValueError(f"Expected shape {(batch_size, 1)}, got {random_angles.shape}")

    angle_intervals = torch.zeros(batch_size, device=device)  # i.e. sample no angles

    random_eyes, random_centers, random_angles = random_fov_eye_space_coordinates(
        heights=heights,
        widths=widths,
        focal_lengths_x=focal_lengths_x,
        focal_lengths_y=focal_lengths_y,
        eye_min_dists=eye_min_dists,
        eye_max_dists=eye_max_dists,
        angle_intervals=angle_intervals,
    )

    if not torch.isclose(random_angles, torch.zeros_like(random_angles)).all():
        raise ValueError(f"Expected all angles to be zero, got {random_angles}")

    eye_max_dists = eye_min_dists / 2.0

    try:
        _, _, _ = random_fov_eye_space_coordinates(
            heights=heights,
            widths=widths,
            focal_lengths_x=focal_lengths_x,
            focal_lengths_y=focal_lengths_y,
            eye_min_dists=eye_min_dists,
            eye_max_dists=eye_max_dists,
            angle_intervals=angle_intervals,
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")


if __name__ == "__main__":
    # test_random_position_in_hollow_spheres()
    test_random_fov_eye_space_coordinates()
