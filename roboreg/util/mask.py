import cv2
import numpy as np
from scipy.signal import convolve2d


def mask_dilate_with_kernel(
    mask: np.ndarray, kernel: np.ndarray = np.ones([10, 10])
) -> np.ndarray:
    extended_mask = convolve2d(mask, kernel, mode="same")
    extended_mask = np.where(extended_mask > 0.0, 255.0, 0.0).astype(np.uint8)
    return extended_mask


def mask_erode_with_kernel(
    mask: np.ndarray, kernel: np.ndarray = np.ones([4, 4])
) -> np.ndarray:
    shrinked_mask = cv2.erode(mask, kernel)
    return shrinked_mask


def mask_extract_boundary(
    mask: np.ndarray,
    erosion_kernel: np.ndarray = np.ones([10, 10]),
) -> np.ndarray:
    boundary_mask = mask - cv2.erode(mask, erosion_kernel)
    return boundary_mask


def mask_extract_extended_boundary(
    mask: np.ndarray,
    erosion_kernel: np.ndarray = np.ones([10, 10]),
    dilation_kernel: np.ndarray = np.ones([10, 10]),
) -> np.ndarray:
    extended_boundary_mask = mask_dilate_with_kernel(
        mask=mask, kernel=dilation_kernel
    ) - mask_erode_with_kernel(mask=mask, kernel=erosion_kernel)
    return extended_boundary_mask


def mask_exponential_distance_transform(
    mask: np.ndarray, sigma: float = 2.0
) -> np.ndarray:
    inverse_mask = np.where(mask > 0.0, 0.0, 1.0).astype(np.uint8)
    distance_map = cv2.distanceTransform(
        inverse_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    distance_map = np.exp(-distance_map / sigma)
    return distance_map
