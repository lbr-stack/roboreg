import numpy as np


def clean_xyz(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r"""

    Args:
        xyz: Point cloud of HxWx3.
        mask: Mask for the point cloud.

    Returns:
        The cleaned point cloud of shape Nx3.
    """
    # mask the cloud
    clean_xyz = np.where(mask[..., None], xyz, np.nan)
    # remove nan
    clean_xyz = clean_xyz[~np.isnan(clean_xyz).any(axis=2)]
    clean_xyz = clean_xyz[~np.isinf(clean_xyz).any(axis=1)]
    return clean_xyz


def sub_sample(data: np.ndarray, N: int) -> np.ndarray:
    if data.shape[0] < N:
        print(
            "N must be smaller than the number of points in data. Using all available."
        )
        N = data.shape[0]
    indices = np.random.choice(data.shape[0], N, replace=False)
    sampled_points = data[indices]
    return sampled_points
