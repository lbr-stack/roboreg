import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import torch

from roboreg.util import (
    random_fov_eye_space_coordinates,
    random_position_in_hollow_sphere,
)


def test_random_position_in_hollow_sphere() -> None:
    inner_radius = -1.0
    outer_radius = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        random_position_in_hollow_sphere(
            inner_radius=inner_radius, outer_radius=outer_radius, device=device
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")

    inner_radius = 1.0
    outer_radius = inner_radius / 2.0

    try:
        random_position_in_hollow_sphere(
            inner_radius=inner_radius, outer_radius=outer_radius, device=device
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")

    inner_radius = 0.0
    outer_radius = 0.0

    try:
        random_position_in_hollow_sphere(
            inner_radius=inner_radius, outer_radius=outer_radius, device=device
        )

    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")

    inner_radius = 0.0
    outer_radius = 1.0
    batch_size = 10

    position = random_position_in_hollow_sphere(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        batch_size=batch_size,
        device=device,
    )

    if position.shape != (batch_size, 3):
        raise ValueError(f"Expected shape {(batch_size, 3)}, got {position.shape}")


def test_random_fov_eye_space_coordinates() -> None:
    height = 1
    width = 1
    focal_length_x = 1.0
    focal_length_y = 1.0
    eye_min_dist = 1.0
    eye_max_dist = 5.0
    angle_interval = torch.pi
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random_eye, random_center, random_angle = random_fov_eye_space_coordinates(
        height=height,
        width=width,
        focal_length_x=focal_length_x,
        focal_length_y=focal_length_y,
        eye_min_dist=eye_min_dist,
        eye_max_dist=eye_max_dist,
        angle_interval=angle_interval,
        batch_size=batch_size,
        device=device,
    )

    if random_eye.shape != (batch_size, 3):
        raise ValueError(f"Expected shape {(batch_size, 3)}, got {random_eye.shape}")
    if random_center.shape != (batch_size, 3):
        raise ValueError(f"Expected shape {(batch_size, 3)}, got {random_center.shape}")
    if random_angle.shape != (batch_size, 1):
        raise ValueError(f"Expected shape {(batch_size, 1)}, got {random_angle.shape}")

    angle_interval = 0  # i.e. sample no angles

    random_eye, random_center, random_angle = random_fov_eye_space_coordinates(
        height=height,
        width=width,
        focal_length_x=focal_length_x,
        focal_length_y=focal_length_y,
        eye_min_dist=eye_min_dist,
        eye_max_dist=eye_max_dist,
        angle_interval=angle_interval,
        batch_size=batch_size,
        device=device,
    )

    if not torch.isclose(random_angle, torch.zeros_like(random_angle)).all():
        raise ValueError(f"Expected all angles to be zero, got {random_angle}")

    eye_max_dist = eye_min_dist / 2.0

    try:
        _, _, _ = random_fov_eye_space_coordinates(
            height=height,
            width=width,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            eye_min_dist=eye_min_dist,
            eye_max_dist=eye_max_dist,
            angle_interval=angle_interval,
            batch_size=batch_size,
            device=device,
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")


if __name__ == "__main__":
    # test_random_position_in_hollow_sphere()
    test_random_fov_eye_space_coordinates()
