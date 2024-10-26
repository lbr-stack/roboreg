import cv2
import numpy as np
from scipy.signal import convolve2d


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
