import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import numpy as np
import torch

from roboreg.differentiable.structs import Camera, TorchMeshContainer, VirtualCamera


def test_torch_mesh_container() -> None:
    # test load simple meshes
    torch_robot_mesh = TorchMeshContainer(
        mesh_paths={
            "link_0": "test/data/lbr_med7/mesh/link_0.stl",
            "link_1": "test/data/lbr_med7/mesh/link_1.stl",
        }
    )
    print(torch_robot_mesh.per_mesh_vertex_count)
    print(torch_robot_mesh.lower_vertex_index_lookup)
    print(torch_robot_mesh.upper_vertex_index_lookup)
    print(torch_robot_mesh.lower_face_index_lookup)
    print(torch_robot_mesh.upper_face_index_lookup)

    # test load visual meshes
    torch_robot_mesh = TorchMeshContainer(
        mesh_paths={
            "link_0": "test/data/lbr_med7/mesh/link_0.dae",
            "link_1": "test/data/lbr_med7/mesh/link_1.dae",
        }
    )
    print(torch_robot_mesh.per_mesh_vertex_count)
    print(torch_robot_mesh.lower_vertex_index_lookup)
    print(torch_robot_mesh.upper_vertex_index_lookup)
    print(torch_robot_mesh.lower_face_index_lookup)
    print(torch_robot_mesh.upper_face_index_lookup)

    n_vertices = torch_robot_mesh.vertices.shape[1]
    print(n_vertices)

    # test taret reduction
    target_reduction = 0.6
    torch_robot_mesh = TorchMeshContainer(
        mesh_paths={
            "link_0": "test/data/lbr_med7/mesh/link_0.dae",
            "link_1": "test/data/lbr_med7/mesh/link_1.dae",
        },
        target_reduction=target_reduction,
    )

    reduced_n_vertices = torch_robot_mesh.vertices.shape[1]
    print(reduced_n_vertices)
    print(1.0 - np.round(reduced_n_vertices / n_vertices, 1))

    if not np.isclose(
        1.0 - np.round(reduced_n_vertices / n_vertices, 1), target_reduction
    ):
        raise ValueError("Expected target reduction")


def test_batched_camera() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = (480, 640)

    # 3x3 intrinsic matrix
    # 4x4 extrinsic matrix
    camera = Camera(
        resolution=resolution,
        device=device,
    )

    if camera.intrinsics.shape != (3, 3):
        raise ValueError(f"Expected shape (3, 3), got {camera.intrinsics.shape}")
    if camera.extrinsics.shape != (4, 4):
        raise ValueError(f"Expected shape (4, 4), got {camera.extrinsics.shape}")
    if camera.ht_optical.shape != (4, 4):
        raise ValueError(f"Expected shape (4, 4), got {camera.ht_optical.shape}")

    # construct invalid dim
    try:
        Camera(
            resolution=resolution,
            intrinsics=torch.zeros(4, 4, device=device),
            extrinsics=torch.zeros(3, 3, device=device),
            device=device,
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")

    try:
        intrinsics = torch.zeros(1, 1, 3, 3, device=device)
        extrinsics = torch.zeros(1, 1, 4, 4, device=device)
        Camera(
            resolution=resolution,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            device=device,
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")

    # batched camera
    intrinsics = torch.zeros(1, 3, 3, device=device)
    extrinsics = torch.zeros(1, 4, 4, device=device)

    camera = Camera(
        resolution=resolution,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        device=device,
    )

    if camera.intrinsics.shape != (1, 3, 3):
        raise ValueError(f"Expected shape (1, 3, 3), got {camera.intrinsics.shape}")
    if camera.extrinsics.shape != (1, 4, 4):
        raise ValueError(f"Expected shape (1, 4, 4), got {camera.extrinsics.shape}")
    if camera.ht_optical.shape != (1, 4, 4):
        raise ValueError(f"Expected shape (1, 4, 4), got {camera.ht_optical.shape}")

    # take a point in homogeneous coordinates of shape Bx4xN and project it
    # using the camera extrinsics / intrinsics
    batch_size = 1
    samples = 100
    shape = (batch_size, 4, samples)
    p = torch.rand(shape, device=camera.device)

    # project the point
    p_prime = camera.extrinsics @ p
    if p_prime.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {p_prime.shape}")

    p_prime = p_prime[:, :3, :] / p_prime[:, 3, :]  # to homogeneous coordinates

    projected_shape = (batch_size, 3, samples)
    p_prime = camera.intrinsics @ p_prime
    if p_prime.shape != projected_shape:
        raise ValueError(f"Expected shape {projected_shape}, got {p_prime.shape}")


def test_batched_virtual_camera() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = (480, 640)

    # default construct (no batch size)
    virtual_camera = VirtualCamera(
        resolution=resolution,
        device=device,
    )

    if virtual_camera.perspective_projection.shape != (4, 4):
        raise ValueError(
            f"Expected shape (3, 4), got {virtual_camera.perspective_projection.shape}"
        )

    # construct with batch size
    intrinsics = torch.zeros(1, 3, 3, device=device)
    virtual_camera = VirtualCamera(
        resolution=resolution,
        intrinsics=intrinsics,
        device=device,
    )

    if virtual_camera.perspective_projection.shape != (1, 4, 4):
        raise ValueError(
            f"Expected shape (1, 4, 4), got {virtual_camera.perspective_projection.shape}"
        )


if __name__ == "__main__":
    test_torch_mesh_container()
    # test_batched_camera()
    # test_batched_virtual_camera()
