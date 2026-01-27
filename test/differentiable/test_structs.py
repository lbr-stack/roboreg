import torch

from roboreg.differentiable.structs import Camera, TorchMeshContainer, VirtualCamera
from roboreg.io import load_meshes


def test_torch_mesh_container() -> None:
    paths = {
        "link_0": "test/assets/lbr_med7/mesh/link_0.stl",
        "link_1": "test/assets/lbr_med7/mesh/link_1.stl",
    }
    device = "cpu"
    container = TorchMeshContainer(
        meshes=load_meshes(
            paths=paths,
        ),
        device=device,
    )

    assert container.names == list(paths.keys()), "Expected same mesh names."
    assert container.vertices.size()[1] == sum(
        list(container.per_mesh_vertex_count.values())
    ), "Expected vertex count to match."
    assert container.device == torch.device(
        device
    ), f"Expected container on '{device}' device."
    assert container.vertices.device == torch.device(
        device
    ), f"Expected vertices on '{device}' device."
    assert container.faces.device == torch.device(
        device
    ), f"Expected faces on '{device}' device."


def test_batched_camera() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = (480, 640)

    # 3x3 intrinsic matrix
    # 4x4 extrinsic matrix
    camera = Camera(
        resolution=resolution,
        device=device,
    )

    assert camera.intrinsics.shape == (
        3,
        3,
    ), f"Expected shape (3, 3), got {camera.intrinsics.shape}"
    assert camera.extrinsics.shape == (
        4,
        4,
    ), f"Expected shape (4, 4), got {camera.extrinsics.shape}"
    assert camera.ht_optical.shape == (
        4,
        4,
    ), f"Expected shape (4, 4), got {camera.ht_optical.shape}"

    # construct invalid dim
    try:
        Camera(
            resolution=resolution,
            intrinsics=torch.zeros(4, 4, device=device),
            extrinsics=torch.zeros(3, 3, device=device),
            device=device,
        )
    except ValueError:
        pass
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
    except ValueError:
        pass
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

    assert camera.intrinsics.shape == (
        1,
        3,
        3,
    ), f"Expected shape (1, 3, 3), got {camera.intrinsics.shape}"
    assert camera.extrinsics.shape == (
        1,
        4,
        4,
    ), f"Expected shape (1, 4, 4), got {camera.extrinsics.shape}"
    assert camera.ht_optical.shape == (
        1,
        4,
        4,
    ), f"Expected shape (1, 4, 4), got {camera.ht_optical.shape}"

    # take a point in homogeneous coordinates of shape Bx4xN and project it
    # using the camera extrinsics / intrinsics
    batch_size = 1
    samples = 100
    shape = (batch_size, 4, samples)
    p = torch.rand(shape, device=camera.device)

    # project the point
    p_prime = camera.extrinsics @ p
    assert p_prime.shape == shape, f"Expected shape {shape}, got {p_prime.shape}"

    p_prime = p_prime[:, :3, :] / p_prime[:, 3, :]  # to homogeneous coordinates

    projected_shape = (batch_size, 3, samples)
    p_prime = camera.intrinsics @ p_prime
    assert (
        p_prime.shape == projected_shape
    ), f"Expected shape {projected_shape}, got {p_prime.shape}"


def test_batched_virtual_camera() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = (480, 640)

    # default construct (no batch size)
    virtual_camera = VirtualCamera(
        resolution=resolution,
        device=device,
    )

    assert virtual_camera.perspective_projection.shape == (
        4,
        4,
    ), f"Expected shape (3, 4), got {virtual_camera.perspective_projection.shape}"

    # construct with batch size
    intrinsics = torch.zeros(1, 3, 3, device=device)
    virtual_camera = VirtualCamera(
        resolution=resolution,
        intrinsics=intrinsics,
        device=device,
    )

    assert virtual_camera.perspective_projection.shape == (
        1,
        4,
        4,
    ), f"Expected shape (1, 4, 4), got {virtual_camera.perspective_projection.shape}"


if __name__ == "__main__":
    import os
    import sys

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_torch_mesh_container()
    test_batched_camera()
    test_batched_virtual_camera()
