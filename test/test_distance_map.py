import cv2

from roboreg.util import distance_transform


def test_distance_map() -> None:
    # given segmentation, compute distance map by any means
    mask = cv2.imread("test/data/high_res/mask_0.png", cv2.IMREAD_GRAYSCALE)

    # install via pip from github
    normalized_distance = distance_transform(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("dist_normalized", normalized_distance)
    cv2.waitKey(0)


if __name__ == "__main__":
    test_distance_map()
