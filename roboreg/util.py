import os
import pathlib
import re
from typing import List, Tuple

import cv2
import numpy as np
import xacro
import yaml
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
    erosion_kernel: np.ndarray = np.ones([10, 10]),
) -> np.ndarray:
    boundary_mask = mask - cv2.erode(mask, erosion_kernel)
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


def logarithmic_asymmetric_distance_transform(mask: np.ndarray) -> np.ndarray:
    r"""Compute the logarithmic asymmetric distance transform.

    Args:
        mask: Binary mask.

    Returns:
        The logarithmic asymmetric distance transform.
    """
    inverse_mask = (mask.max() - mask).astype(np.uint8)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_inverse = cv2.distanceTransform(inverse_mask, cv2.DIST_L2, 3)

    dist = np.log(dist + 1.0)

    dist_asymmetric = dist + dist_inverse
    return dist_asymmetric


def normalized_distance_transform(mask: np.ndarray) -> np.ndarray:
    r"""Compute the normalized distance transform.

    Args:
        mask: Binary mask.

    Returns:
        The normalized distance transform.
    """
    dist = cv2.distanceTransform(mask.max() - mask, cv2.DIST_L2, 3)
    dist_normalized = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    return dist_normalized


def normalized_symmetric_distance_function(mask: np.ndarray) -> np.ndarray:
    r"""Commpute the normalized symmetric distance function.
    The function puts zeros at the boundary of the mask.
    The values increase by distance from the boundary.

    Args:
        mask: Binary mask.

    Returns:
        The normalized symmetric distance function.
    """
    inverse_mask = (mask.max() - mask).astype(np.uint8)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_inverse = cv2.distanceTransform(inverse_mask, cv2.DIST_L2, 3)

    dist_symmetric = dist + dist_inverse
    dist_symmetric_normalized = cv2.normalize(
        dist_symmetric, None, 0, 1.0, cv2.NORM_MINMAX
    )
    return dist_symmetric_normalized


def parse_camera_info(camera_info_file: str) -> Tuple[int, int, np.ndarray]:
    r"""Parse camera info file.

    Args:
        camera_info_file (str): Absolute path to the camera info file.

    Returns:
        height (int): Height of the image.
        width (int): Width of the image.
        intrinsic_matrix (np.ndarray): Intrinsic matrix of shape 3x3.
    """
    with open(camera_info_file, "r") as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    height = camera_info["height"]
    width = camera_info["width"]
    if (
        camera_info["camera_matrix"]["cols"] != 3
        or camera_info["camera_matrix"]["rows"] != 3
    ):
        raise ValueError("Camera matrix must be 3x3.")
    intrinsic_matrix = np.array(camera_info["camera_matrix"]["data"]).reshape(3, 3)
    return height, width, intrinsic_matrix


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    mode: str = "r",
    alpha: float = 0.5,
    scale: float = 2.0,
) -> np.ndarray:
    r"""Overlay mask on image.

    Args:
        img: Image of shape HxWx3.
        mask: Mask of shape HxW.
        mode: Color mode. "r", "g", or "b".
        alpha: Alpha value for the mask.
        scale: Scale factor for the image.

    Returns:
        Mask overlayed on image.
    """
    colored_mask = None
    if mode == "r":
        colored_mask = np.stack(
            [np.zeros_like(mask), np.zeros_like(mask), mask], axis=2
        )
    elif mode == "g":
        colored_mask = np.stack(
            [np.zeros_like(mask), mask, np.zeros_like(mask)], axis=2
        )
    elif mode == "b":
        colored_mask = np.stack(
            [mask, np.zeros_like(mask), np.zeros_like(mask)], axis=2
        )
    else:
        raise ValueError("Mode must be r, g, or b.")

    overlay_img_mask = cv2.addWeighted(img, alpha, colored_mask, 1.0, 0)
    # resize by scale
    overlay_img_mask = cv2.resize(
        overlay_img_mask,
        [int(size * scale) for size in overlay_img_mask.shape[:2][::-1]],
    )
    return overlay_img_mask


def find_files(path: str, pattern: str = "img_*.png") -> List[str]:
    r"""Find files in a directory.

    Args:
        path: Path to the directory.
        pattern: Pattern to match.

    Returns:
        List of file names.
    """

    def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(l, key=alphanum_key)

    path = pathlib.Path(path)
    image_paths = list(path.glob(pattern))
    return sorted([image_path.name for image_path in image_paths], key=natural_sort)
