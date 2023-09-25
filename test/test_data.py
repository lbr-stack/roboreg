import cv2
import numpy as np


def test_data() -> None:
    mask = cv2.imread("test/data/low_res/mask_1.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("test/data/low_res/img_1.png", cv2.IMREAD_GRAYSCALE)
    concat = np.concatenate([img, mask, np.where(mask, img, 0)], axis=1)
    cv2.imshow("concat", concat)
    cv2.waitKey()


if __name__ == "__main__":
    test_data()
