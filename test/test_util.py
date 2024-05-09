import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2

from roboreg.util import (
    extend_mask,
    find_files,
    mask_boundary,
    overlay_mask,
    shrink_mask,
)


def test_extend_mask() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/data/lbr_med7/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE
    )
    extended_mask = extend_mask(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("extended_mask", extended_mask)
    cv2.waitKey()


def test_mask_boundary() -> None:
    idx = 1
    img = cv2.imread(f"test/data/lbr_med7/high_res/img_{idx}.png")
    mask = cv2.imread(
        f"test/data/lbr_med7/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE
    )
    boundary_mask = mask_boundary(mask)
    overlay = overlay_mask(img, boundary_mask, mode="b", alpha=1.0, scale=1.0)
    cv2.imshow("mask", mask)
    cv2.imshow("boundary_mask", boundary_mask)
    cv2.imshow("overlay", overlay)
    cv2.waitKey()


def test_shrink_mask() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/data/lbr_med7/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE
    )
    shrinked_mask = shrink_mask(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("shrinked_mask", shrinked_mask)
    cv2.waitKey()


def test_find_files() -> None:
    path = "test/data/lbr_med7/high_res"
    for mask_file in find_files(path, "mask_*.png"):
        print(mask_file)


if __name__ == "__main__":
    # test_extend_mask()
    test_mask_boundary()
    # test_shrink_mask()
    # test_find_files()
