import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import numpy as np
import torch

from roboreg.util.points import clean_xyz, compute_vertex_normals, to_homogeneous


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
    test_compute_vertex_normals()
