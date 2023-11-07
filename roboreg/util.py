import os

import cv2
import numpy as np
import xacro
from ament_index_python import get_package_share_directory
from scipy.signal import convolve2d

from roboreg.o3d_robot import O3DRobot


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


def sub_sample(data: np.ndarray, N: int) -> np.ndarray:
    if data.shape[0] < N:
        print(
            "N must be smaller than the number of points in data. Using all available."
        )
        N = data.shape[0]
    indices = np.random.choice(data.shape[0], N, replace=False)
    sampled_points = data[indices]
    return sampled_points


def extend_mask(mask: np.ndarray, kernel: np.ndarray = np.ones([10, 10])) -> np.ndarray:
    extended_mask = convolve2d(mask, kernel, mode="same")
    extended_mask = np.where(extended_mask > 0.0, 255.0, 0.0).astype(np.uint8)
    return extended_mask


def shrink_mask(mask: np.ndarray, kernel: np.ndarray = np.ones([4, 4])) -> np.ndarray:
    shrinked_mask = cv2.erode(mask, kernel)
    return shrinked_mask


def mask_boundary(
    mask: np.ndarray,
    dilation_kernel: np.ndarray = np.ones([1, 1]),
    erosion_kernel: np.ndarray = np.ones([20, 20]),
) -> np.ndarray:
    boundary_mask = cv2.dilate(mask, dilation_kernel) - cv2.erode(mask, erosion_kernel)
    return boundary_mask


def generate_o3d_robot(
    package_name: str = "lbr_description",
    relative_xacro_path: str = "urdf/med7/med7.urdf.xacro",
) -> O3DRobot:
    # load robot
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory(package_name),
            relative_xacro_path,
        )
    )
    robot = O3DRobot(urdf=urdf)
    return robot
