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
    compute_vertex_normals,
    depth_to_xyz,
    from_homogeneous,
    generate_ht_optical,
    to_homogeneous,
)


def test_clean_xyz() -> None:
    # no batch dimension
    points = np.random.rand(256, 256, 3)
    mask = np.random.rand(256, 256)
    mask = mask > 0.5

    clean_points = clean_xyz(xyz=points, mask=mask)
    if clean_points.shape[0] != mask.sum():
        raise ValueError(
            f"Expected {mask.sum()} points, got {clean_points.shape[0]} points."
        )
    if clean_points.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {clean_points.shape[1]} channels.")


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
    xzys = to_homogeneous(xyzs)
    ht_optical = generate_ht_optical(xyzs.shape[0], dtype=torch.float32, device=device)
    xyzs = torch.matmul(xzys, ht_optical.transpose(-1, -2))
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


def test_compute_vertex_normals() -> None:
    n_vertices = 100
    n_faces = 120
    vertices = torch.rand(n_vertices, 3)
    faces = torch.randint(0, n_vertices, (n_faces, 3))  # random faces

    vertex_normals = compute_vertex_normals(vertices=vertices, faces=faces)

    if vertex_normals.shape[0] != n_vertices:
        raise ValueError(
            f"Expected {n_vertices} normals, got {vertex_normals.shape[0]} normals."
        )
    if vertex_normals.shape[1] != 3:
        raise ValueError(
            f"Expected 3 channels, got {vertex_normals.shape[1]} channels."
        )

    # test homogeneous coordinates
    vertices = to_homogeneous(vertices)
    vertex_normals = compute_vertex_normals(vertices=vertices, faces=faces)

    if vertex_normals.shape[0] != n_vertices:
        raise ValueError(
            f"Expected {n_vertices} normals, got {vertex_normals.shape[0]} normals."
        )
    if vertex_normals.shape[1] != 4:
        raise ValueError(
            f"Expected 3 channels, got {vertex_normals.shape[1]} channels."
        )

    # test for a very simple mesh
    vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    vertex_normals = compute_vertex_normals(vertices=vertices, faces=faces)

    if not torch.allclose(
        vertex_normals,
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    ):
        raise ValueError("Expected all normals to be [0, 0, 1].")

    # test for a very simple mesh with homogeneous coordinates
    vertices = to_homogeneous(vertices)
    vertex_normals = compute_vertex_normals(vertices=vertices, faces=faces)

    if not torch.allclose(
        vertex_normals,
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    ):
        raise ValueError("Expected all normals to be [0, 0, 1, 1].")


if __name__ == "__main__":
    # test_clean_xyz()
    # test_depth_to_xyz()
    test_realsense_depth_to_xyz()
    # test_compute_vertex_normals()
