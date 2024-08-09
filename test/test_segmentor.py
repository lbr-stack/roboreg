import os

import cv2
import numpy as np
import torch

from roboreg.detector import OpenCVDetector
from roboreg.segmentor import Sam2Segmentor, SamSegmentor


def test_sam2_segmentor() -> None:
    img = cv2.imread("test/data/lbr_med7/zed2i/high_res/image_1.png")

    # detect
    detector = OpenCVDetector(n_positive_samples=5)  # number of detected samples
    samples, labels = detector.detect(img)

    # segment
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segmentor = Sam2Segmentor(device=device)
    mask = segmentor(img, np.array(samples), np.array(labels))

    # visualize
    cv2.imshow("masked_img", np.where(np.expand_dims(mask, -1), img, 0))
    cv2.waitKey()


def test_sam_segmentor() -> None:
    img = cv2.imread("test/data/lbr_med7/zed2i/high_res/image_1.png")

    # detect
    detector = OpenCVDetector(n_positive_samples=5)  # number of detected samples
    samples, labels = detector.detect(img)

    # segment
    checkpoint = os.path.join(
        os.environ["HOME"],
        "Downloads/sam_checkpoints/sam_vit_h_4b8939.pth",
    )
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segmentor = SamSegmentor(
        checkpoint=checkpoint, model_type=model_type, device=device
    )
    mask = segmentor(img, np.array(samples), np.array(labels))

    # visualize
    cv2.imshow("masked_img", np.where(np.expand_dims(mask, -1), img, 0))
    cv2.waitKey()


if __name__ == "__main__":
    test_sam2_segmentor()
    # test_sam_segmentor()
