import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import cv2
import numpy as np
import pytest

from roboreg.util import (
    mask_dilate_with_kernel,
    mask_distance_transform,
    mask_erode_with_kernel,
    mask_exponential_decay,
    mask_extract_boundary,
    mask_extract_extended_boundary,
    overlay_mask,
)


@pytest.mark.skip(reason="To be fixed.")
def test_dilate_with_kernel() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/assets/lbr_med7_r800/samples/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    dilated_mask = mask_dilate_with_kernel(mask)
    cv2.imwrite("test_dilate_with_kernel.mask.png", mask)
    cv2.imwrite("test_dilate_with_kernel.dilated_mask.png", dilated_mask)


@pytest.mark.skip(reason="To be fixed.")
def test_distance_transform() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/assets/lbr_med7_r800/samples/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )

    # show distance map
    distance_map = mask_distance_transform(mask)
    distance_map = (distance_map / distance_map.max() * 255.0).astype(
        np.uint8
    )  # normalize for visualization
    cv2.imwrite("test_distance_transform.mask.png", mask)
    cv2.imwrite("test_distance_transform.distance_map.png", distance_map)

    # show inverse distance map
    inverse_mask = np.where(mask > 0, 0, 255).astype(np.uint8)
    inverse_distance_map = mask_distance_transform(inverse_mask)
    inverse_distance_map = (
        inverse_distance_map / inverse_distance_map.max() * 255.0
    ).astype(
        np.uint8
    )  # normalize for visualization
    cv2.imwrite("test_distance_transform.inverse_mask.png", inverse_mask)
    cv2.imwrite(
        "test_distance_transform.inverse_distance_map.png", inverse_distance_map
    )


@pytest.mark.skip(reason="To be fixed.")
def test_erode_with_kernel() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/assets/lbr_med7_r800/samples/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    eroded_mask = mask_erode_with_kernel(mask)
    cv2.imwrite("test_erode_with_kernel.mask.png", mask)
    cv2.imwrite("test_erode_with_kernel.eroded_mask.png", eroded_mask)


@pytest.mark.skip(reason="To be fixed.")
def test_exponential_decay() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/assets/lbr_med7_r800/samples/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    exponential_decay = mask_exponential_decay(mask)
    cv2.imwrite("test_exponential_decay.mask.png", mask)
    cv2.imwrite("test_exponential_decay.exponential_decay.png", exponential_decay)


@pytest.mark.skip(reason="To be fixed.")
def test_extract_boundary() -> None:
    idx = 1
    img = cv2.imread(f"test/assets/lbr_med7_r800/samples/left_image_{idx}.png")
    mask = cv2.imread(
        f"test/assets/lbr_med7_r800/samples/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    boundary_mask = mask_extract_boundary(mask)
    overlay = overlay_mask(img, boundary_mask, mode="b", alpha=1.0, scale=1.0)
    cv2.imwrite("test_extract_boundary.mask.png", mask)
    cv2.imwrite("test_extract_boundary.boundary_mask.png", boundary_mask)
    cv2.imwrite("test_extract_boundary.overlay.png", overlay)


@pytest.mark.skip(reason="To be fixed.")
def test_extract_extended_boundary() -> None:
    idx = 1
    img = cv2.imread(f"test/assets/lbr_med7_r800/samples/left_image_{idx}.png")
    mask = cv2.imread(
        f"test/assets/lbr_med7_r800/samples/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    extended_boundary_mask = mask_extract_extended_boundary(
        mask, dilation_kernel=np.ones([2, 2]), erosion_kernel=np.ones([10, 10])
    )
    overlay = overlay_mask(img, extended_boundary_mask, mode="b", alpha=1.0, scale=1.0)
    cv2.imwrite("test_extract_extended_boundary.mask.png", mask)
    cv2.imwrite(
        "test_extract_extended_boundary.extended_boundary_mask.png",
        extended_boundary_mask,
    )
    cv2.imwrite("test_extract_extended_boundary.overlay.png", overlay)


if __name__ == "__main__":
    test_dilate_with_kernel()
    test_distance_transform()
    test_erode_with_kernel()
    test_exponential_decay()
    test_extract_boundary()
    test_extract_extended_boundary()
