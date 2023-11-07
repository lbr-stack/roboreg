import cv2

from roboreg.util import extend_mask, mask_boundary, shrink_mask


def test_extend_mask() -> None:
    mask = cv2.imread("test/data/low_res/mask_0.png", cv2.IMREAD_GRAYSCALE)
    extended_mask = extend_mask(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("extended_mask", extended_mask)
    cv2.waitKey()


def test_mask_boundary() -> None:
    mask = cv2.imread("test/data/low_res/mask_0.png", cv2.IMREAD_GRAYSCALE)
    boundary_mask = mask_boundary(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("boundary_mask", boundary_mask)
    cv2.waitKey()


def test_shrink_mask() -> None:
    mask = cv2.imread("test/data/low_res/mask_0.png", cv2.IMREAD_GRAYSCALE)
    shrinked_mask = shrink_mask(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("shrinked_mask", shrinked_mask)
    cv2.waitKey()


if __name__ == "__main__":
    # test_extend_mask()
    test_mask_boundary()
    # test_shrink_mask()
