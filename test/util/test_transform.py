import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import numpy as np
import torch

from roboreg.io import find_files, parse_camera_info
from roboreg.util import (
    clean_xyz,
    depth_to_xyz,
    from_homogeneous,
    generate_ht_optical,
    look_at_from_angle,
    to_homogeneous,
)


def test_depth_to_xyz() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height, width = 256, 256
    intrinsics = torch.tensor(
        [
            [height, 0.0, height / 2],
            [0.0, width, width / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    depth = torch.ones(height, width, device=device)

    xyz = depth_to_xyz(depth=depth, intrinsics=intrinsics)

    if xyz.shape[0] != height:
        raise ValueError(f"Expected {height} points, got {xyz.shape[0]} points.")
    if xyz.shape[1] != width:
        raise ValueError(f"Expected {width} points, got {xyz.shape[1]} points.")
    if xyz.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got {xyz.shape[2]} channels.")

    # test batched depth
    batch_size = 4
    depth = depth.repeat(batch_size, 1, 1)

    xyz = depth_to_xyz(depth=depth, intrinsics=intrinsics)

    if xyz.shape[0] != batch_size:
        raise ValueError(f"Expected {batch_size} points, got {xyz.shape[0]} points.")
    if xyz.shape[1] != height:
        raise ValueError(f"Expected {height} points, got {xyz.shape[1]} points.")
    if xyz.shape[2] != width:
        raise ValueError(f"Expected {width} points, got {xyz.shape[2]} points.")
    if xyz.shape[3] != 3:
        raise ValueError(f"Expected 3 channels, got {xyz.shape[3]} channels.")

    # test batched all
    intrinsics = intrinsics.repeat(batch_size, 1, 1)

    xyz = depth_to_xyz(depth=depth, intrinsics=intrinsics)

    if xyz.shape[0] != batch_size:
        raise ValueError(f"Expected {batch_size} points, got {xyz.shape[0]} points.")
    if xyz.shape[1] != height:
        raise ValueError(f"Expected {height} points, got {xyz.shape[1]} points.")
    if xyz.shape[2] != width:
        raise ValueError(f"Expected {width} points, got {xyz.shape[2]} points.")
    if xyz.shape[3] != 3:
        raise ValueError(f"Expected 3 channels, got {xyz.shape[3]} channels.")


def test_realsense_depth_to_xyz() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "test/data/xarm/realsense"
    depth_files = find_files(path, "depth_*.npy")
    height, width, intrinsics = parse_camera_info(
        "test/data/xarm/realsense/camera_info.yaml"
    )
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    depths = torch.tensor(
        [np.load(os.path.join(path, depth_file)) for depth_file in depth_files],
        dtype=torch.float32,
        device=device,
    )
    xyzs = depth_to_xyz(depth=depths, intrinsics=intrinsics, z_max=1.5)

    # flatten BxHxWx3 -> Bx(H*W)x3
    xyzs = xyzs.view(-1, height * width, 3)
    xyzs = to_homogeneous(xyzs)
    ht_optical = generate_ht_optical(xyzs.shape[0], dtype=torch.float32, device=device)
    xyzs = torch.matmul(xyzs, ht_optical.transpose(-1, -2))
    xyzs = from_homogeneous(xyzs)

    # unflatten
    xyzs = xyzs.view(-1, height, width, 3)
    xyzs = xyzs.cpu().numpy()

    for idx, depth_file in enumerate(depth_files):
        np.save(
            os.path.join(path, f"xyz_{depth_file.split('_')[-1].split('.')[0]}.npy"),
            xyzs[idx],
        )

    def test_display_xyz() -> None:
        import pyvista as pv

        xyz_files = find_files(path, "xyz_*.npy")
        xyzs = [
            np.load(os.path.join(path, xyz_files)).reshape(
                -1, 3
            )  # flatten HxWx3 -> (H*W)x3
            for xyz_files in xyz_files
        ]
        xyzs = np.concatenate(xyzs, axis=0)
        xyzs = clean_xyz(xyz=xyzs)

        pl = pv.Plotter()
        pl.background_color = [0, 0, 0]
        pl.add_axes()
        pl.add_points(xyzs, scalars=xyzs[:, 2] / xyzs[:, 2].max(), point_size=1)
        pl.show()

    test_display_xyz()


def test_look_at_from_angle() -> None:
    batch_size = 4
    eye = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).unsqueeze(0)
    center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0)
    angle = torch.tensor([np.pi], dtype=torch.float32).unsqueeze(0)
    eye = eye.repeat(batch_size, 1)
    center = center.repeat(batch_size, 1)
    angle = angle.repeat(batch_size, 1)

    ht = look_at_from_angle(
        eye=eye,
        center=center,
        angle=angle,
    )

    if angle.shape != (batch_size, 1):  # check this remains unchanged
        raise ValueError(f"Expected shape ({batch_size}, 1), got {angle.shape}.")

    if ht.shape != (batch_size, 4, 4):
        raise ValueError(f"Expected shape ({batch_size}, 4, 4), got {ht.shape}.")


if __name__ == "__main__":
    test_depth_to_xyz()
    test_realsense_depth_to_xyz()
    test_look_at_from_angle()
