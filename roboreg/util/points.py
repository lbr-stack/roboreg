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
