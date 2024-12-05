import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import cv2

from roboreg.util import (
    mask_dilate_with_kernel,
    mask_erode_with_kernel,
    mask_exponential_distance_transform,
    mask_extract_boundary,
    overlay_mask,
)


def test_dilate_with_kernel() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/data/lbr_med7/zed2i/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE
    )
    extended_mask = mask_dilate_with_kernel(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("extended_mask", extended_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_erode_with_kernel() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/data/lbr_med7/zed2i/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE
    )
    shrinked_mask = mask_erode_with_kernel(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("shrinked_mask", shrinked_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_exponential_distance_transform() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/data/lbr_med7/zed2i/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE
    )
    exponential_distance_map = mask_exponential_distance_transform(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("exponential_distance_map", exponential_distance_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_extract_boundary() -> None:
    idx = 1
    img = cv2.imread(f"test/data/lbr_med7/zed2i/high_res/image_{idx}.png")
    mask = cv2.imread(
        f"test/data/lbr_med7/zed2i/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE
    )
    boundary_mask = mask_extract_boundary(mask)
    overlay = overlay_mask(img, boundary_mask, mode="b", alpha=1.0, scale=1.0)
    cv2.imshow("mask", mask)
    cv2.imshow("boundary_mask", boundary_mask)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # test_dilate_with_kernel()
    # test_erode_with_kernel()
    test_exponential_distance_transform()
    # test_extract_boundary()
