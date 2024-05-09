import os

import cv2
import numpy as np

from roboreg.detector import OpenCVDetector
from roboreg.segmentor import SamSegmentor


def test_sam_segmentor() -> None:
    img = cv2.imread("test/data/lbr_med7/low_res/img_1.png")

    # detect
    detector = OpenCVDetector(buffer_size=5)  # number of detected points
    points, labels = detector.detect(img)

    # segment
    sam_checkpoint = os.path.join(
        os.environ["HOME"],
        "Downloads/segment_anything_checkpoints/sam_vit_h_4b8939.pth",
    )
    model_type = "vit_h"
    device = "cuda"

    segmentor = SamSegmentor(
        sam_checkpoint=sam_checkpoint, model_type=model_type, device=device
    )
    mask = segmentor(img, np.array(points), np.array(labels))

    # visualize
    cv2.imshow("masked_img", np.where(np.expand_dims(mask, -1), img, 0))
    cv2.waitKey()


if __name__ == "__main__":
    test_sam_segmentor()
