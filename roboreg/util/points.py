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
    if xyz.shape[:2] != mask.shape:
        raise ValueError("Expected xyz and mask to have the same spatial dimensions.")
    if xyz.shape[-1] != 3:
        raise ValueError("Expected xyz to have 3 channels.")
    if mask is not None:
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
