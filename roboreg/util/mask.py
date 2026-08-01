import cv2
import numpy as np


def _as_binary_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _as_uint8_kernel(kernel: np.ndarray) -> np.ndarray:
    if kernel.ndim != 2:
        raise ValueError(f"Expected a 2D kernel, got shape {kernel.shape}.")

    return np.where(kernel > 0, 1, 0).astype(np.uint8)


def mask_dilate_with_kernel(
    mask: np.ndarray,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    if kernel is None:
        kernel = np.ones((10, 10), dtype=np.uint8)

    mask = _as_binary_uint8(mask)
    kernel = _as_uint8_kernel(kernel)

    return cv2.dilate(mask, kernel)


def mask_distance_transform(mask: np.ndarray) -> np.ndarray:
    mask = _as_binary_uint8(mask)

    return cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def mask_erode_with_kernel(
    mask: np.ndarray,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    if kernel is None:
        kernel = np.ones((4, 4), dtype=np.uint8)

    mask = _as_binary_uint8(mask)
    kernel = _as_uint8_kernel(kernel)

    return cv2.erode(mask, kernel)


def mask_exponential_decay(
    mask: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    mask = _as_binary_uint8(mask)
    inverse_mask = cv2.bitwise_not(mask)

    distance_map = mask_distance_transform(inverse_mask)

    return np.exp(-distance_map / sigma).astype(np.float32)


def mask_extract_boundary(
    mask: np.ndarray,
    erosion_kernel: np.ndarray | None = None,
) -> np.ndarray:
    if erosion_kernel is None:
        erosion_kernel = np.ones((10, 10), dtype=np.uint8)

    mask = _as_binary_uint8(mask)
    eroded_mask = mask_erode_with_kernel(
        mask=mask,
        kernel=erosion_kernel,
    )

    return cv2.subtract(mask, eroded_mask)


def mask_extract_extended_boundary(
    mask: np.ndarray,
    dilation_kernel: np.ndarray | None = None,
    erosion_kernel: np.ndarray | None = None,
) -> np.ndarray:
    if dilation_kernel is None:
        dilation_kernel = np.ones((10, 10), dtype=np.uint8)

    if erosion_kernel is None:
        erosion_kernel = np.ones((10, 10), dtype=np.uint8)

    dilated_mask = mask_dilate_with_kernel(
        mask=mask,
        kernel=dilation_kernel,
    )

    eroded_mask = mask_erode_with_kernel(
        mask=mask,
        kernel=erosion_kernel,
    )

    return cv2.subtract(dilated_mask, eroded_mask)
