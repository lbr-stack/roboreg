import cv2
import numpy as np
import pytest
import torch

from roboreg.detector import OpenCVDetector
from roboreg.segmentor import Sam2Segmentor


@pytest.mark.skip(reason="To be fixed.")
def test_sam2_segmentor() -> None:
    img = cv2.imread("test/assets/lbr_med7_r800/samples/left_image_1.png")

    # detect
    detector = OpenCVDetector(n_positive_samples=5)  # number of detected samples
    samples, labels = detector.detect(img)

    # segment
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segmentor = Sam2Segmentor(device=device)
    p = segmentor(img, np.array(samples), np.array(labels))

    # visualize
    cv2.imshow(
        "masked_img",
        np.where(np.expand_dims(p > segmentor.pth, -1), img, 0),
    )
    cv2.imshow("probability", p)
    cv2.waitKey(0)


if __name__ == "__main__":
    test_sam2_segmentor()
