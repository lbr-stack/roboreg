from typing import Optional

import numpy as np
import torch


def clean_xyz(xyz: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    r"""Masks a point cloud and removes invalid points.

    Args:
        xyz: Point cloud of shape HxWx3.
        mask: Mask for the point cloud of shape HxW.

    Returns:
        Cleaned and flattened point cloud of shape Nx3.
    """
    if xyz.shape[-1] != 3:
        raise ValueError("Expected xyz to have 3 channels.")
    if mask is not None:
        if xyz.shape[:2] != mask.shape:
            raise ValueError(
                "Expected xyz and mask to have the same spatial dimensions."
            )
        # mask the cloud
        clean_xyz = np.where(mask[..., None], xyz, np.nan)
    else:
        clean_xyz = xyz
    # remove nans and infs
    clean_xyz = clean_xyz[~np.isnan(clean_xyz).any(axis=-1)]
    clean_xyz = clean_xyz[~np.isinf(clean_xyz).any(axis=-1)]
    return clean_xyz


def depth_to_xyz(
    depth: torch.FloatTensor,
    intrinsics: torch.FloatTensor,
    z_min: float = 0.01,
    z_max: float = 2,
    conversion_factor: float = 1.0e3,
) -> torch.FloatTensor:
    r"""Converts a depth image to a point cloud. Note that this function uses the OpenCV convention.

    Args:
        depth (torch.FloatTensor): Depth image of shape HxW / BxHxW / Bx1xHxW.
        intrinsics (torch.FloatTensor): Camera intrinsics of shape 3x3 or Bx3x3.
        z_min (float): Minimum depth value.
        z_max (float): Maximum depth value.
        conversion_factor (float): Conversion factor for depth. Computes z = depth / conversion_factor.

    Returns:
        Point cloud of shape HxWx3 or BxHxWx3.
    """
    if intrinsics.dim() > depth.dim():
        raise ValueError("Expected intrinsics to have fewer dimensions than depth.")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("Expected intrinsics to have shape 3x3 or Bx3x3.")
    if z_min >= z_max:
        raise ValueError("Expected z_min to be less than z_max.")
    if conversion_factor <= 0:
        raise ValueError("Expected conversion_factor to be greater than zero.")
    if z_min <= 0:
        raise ValueError("Expected z_min to be greater than zero.")
    height, width = depth.shape[-2:]
    x = torch.linspace(0, width - 1, width, device=depth.device)
    y = torch.linspace(0, height - 1, height, device=depth.device)
    y, x = torch.meshgrid(y, x, indexing="ij")
    if depth.dim() == 3:
        x = x.unsqueeze(0).expand(depth.shape[0], -1, -1)
        y = y.unsqueeze(0).expand(depth.shape[0], -1, -1)
    z = depth / conversion_factor
    # fill with nans where z_min <= z <= z_max
    z = torch.where((z < z_min) | (z > z_max), torch.nan, z)
    if intrinsics.dim() == 2 and depth.dim() == 3:
        intrinsics = intrinsics.unsqueeze(0).expand(depth.shape[0], -1, -1)
    x = (
        (x - intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1))
        * z
        / intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    )
    y = (
        (y - intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1))
        * z
        / intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    )
    return torch.stack((x, y, z), dim=-1)


def generate_ht_optical(
    batch_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = "cuda",
) -> torch.Tensor:
    ht_optical = torch.zeros(4, 4, dtype=dtype, device=device)
    if batch_size is not None:
        ht_optical = ht_optical.unsqueeze(0).expand(batch_size, -1, -1)
    ht_optical[..., 0, 2] = (
        1.0  # OpenCV-oriented optical frame, in quaternions: [0.5, -0.5, 0.5, -0.5] (w, x, y, z)
    )
    ht_optical[..., 1, 0] = -1.0
    ht_optical[..., 2, 1] = -1.0
    ht_optical[..., 3, 3] = 1.0
    return ht_optical


def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N) to (..., N+1) by appending ones."""
    return torch.nn.functional.pad(x, (0, 1), "constant", 1.0)


def from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N+1) to (..., N)."""
    return x[..., :-1]


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    r"""Compute vertex normals. Currently doesn't support a batch dimension.

    Args:
        vertices (torch.Tensor): Vertices of shape (N, 3) or (N, 4) in homogeneous coordinates.
        faces (torch.Tensor): Faces of shape (M, 3).

    Returns:
        Vertex normals of shape (N, 3).
    """
    face_normals = torch.nn.functional.normalize(
        torch.cross(
            vertices[faces[:, 1], :3] - vertices[faces[:, 0], :3],
            vertices[faces[:, 2], :3] - vertices[faces[:, 0], :3],
            dim=-1,
        ),
        p=2,
        dim=-1,
    )

    vertex_normals = torch.zeros_like(vertices)

    if vertex_normals.shape[-1] == 4:
        vertex_normals[:, 3] = 1.0

    # accumulate face normals to vertices using scatter_add
    vertex_normals[:, :3] = vertex_normals[:, :3].index_add(
        0, faces.view(-1), face_normals.repeat(1, 3).view(-1, 3)
    )

    # normalize vertex normals
    vertex_normals[:, :3] = torch.nn.functional.normalize(
        vertex_normals[:, :3], p=2, dim=-1
    )
    return vertex_normals
