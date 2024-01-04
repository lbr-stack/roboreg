import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2

from roboreg.util import (
    normalized_distance_transform,
    normalized_symmetric_distance_function,
)


def test_normalized_distance_map() -> None:
    # given segmentation, compute distance map by any means
    mask = cv2.imread("test/data/high_res/mask_0.png", cv2.IMREAD_GRAYSCALE)

    # install via pip from github
    normalized_distance = normalized_distance_transform(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("dist_normalized", normalized_distance)
    cv2.waitKey(0)


def test_normalized_symmetric_distance_map() -> None:
    # given segmentation, compute distance map by any means
    mask = cv2.imread("test/data/high_res/mask_0.png", cv2.IMREAD_GRAYSCALE)

    # install via pip from github
    dist_symmetric = normalized_symmetric_distance_function(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("dist_symmetric_normalized", dist_symmetric)
    cv2.waitKey(0)


if __name__ == "__main__":
    # test_normalized_distance_map()
    test_normalized_symmetric_distance_map()
