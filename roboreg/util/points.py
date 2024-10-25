import numpy as np
import torch


def clean_xyz(xyz: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    r"""

    Args:
        xyz: Point cloud of HxWx3.
        mask: Mask for the point cloud.

    Returns:
        The cleaned point cloud of shape Nx3.
    """
    if mask is not None:
        # mask the cloud
        clean_xyz = np.where(mask[..., None], xyz, np.nan)
    else:
        clean_xyz = xyz
    # remove nan
    clean_xyz = clean_xyz[~np.isnan(clean_xyz).any(axis=2)]
    clean_xyz = clean_xyz[~np.isinf(clean_xyz).any(axis=1)]
    return clean_xyz


def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N) to (..., N+1) by appending ones."""
    return torch.nn.functional.pad(x, (0, 1), "constant", 1.0)


def from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N+1) to (..., N)."""
    return x[..., :-1]
