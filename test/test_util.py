import cv2

from roboreg.util import extend_mask


def test_extend_mask() -> None:
    mask = cv2.imread("test/data/low_res/mask_0.png", cv2.IMREAD_GRAYSCALE)
    extended_mask = extend_mask(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("extended_mask", extended_mask)
    cv2.waitKey()


if __name__ == "__main__":
    test_extend_mask()
