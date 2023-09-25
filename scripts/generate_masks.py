import argparse
import os
import pathlib

import cv2
import numpy as np

from roboreg.detector import OpenCVDetector
from roboreg.segmentor import SamSegmentor

from scripts.common import find_files


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="img_*.png")
    parser.add_argument("--buffer_size", type=int, default=5)
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="FUll path to SAM checkpoint. Should be named ~sam_vit_h_4b8939.pth",
    )

    return parser.parse_args()


def main():
    args = args_factory()
    path = pathlib.Path(args.path)
    image_names = find_files(path.absolute(), args.pattern)

    # detect
    detector = OpenCVDetector(buffer_size=args.buffer_size)  # number of detected points

    # segment
    sam_checkpoint = os.path.join(args.sam_checkpoint)
    model_type = "vit_h"
    device = "cuda"

    segmentor = SamSegmentor(
        sam_checkpoint=sam_checkpoint, model_type=model_type, device=device
    )

    for image_name in image_names:
        img = cv2.imread(os.path.join(path.absolute(), image_name))
        points, labels = detector.detect(img)
        detector.clear()
        mask = segmentor(img, np.array(points), np.array(labels))

        # write mask
        mask_path = os.path.join(path.absolute(), image_name.replace("img", "mask"))
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    main()
