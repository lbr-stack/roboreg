import argparse
import os
import pathlib

import cv2
import numpy as np
from common import find_files


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--image_pattern", type=str, default="img_*.png")
    parser.add_argument("--mask_pattern", type=str, default="mask_*.png")

    return parser.parse_args()


def main():
    args = args_factory()
    path = pathlib.Path(args.path)
    image_names = find_files(path.absolute(), args.image_pattern)
    mask_names = find_files(path.absolute(), args.mask_pattern)

    for image_name, mask_name in zip(image_names, mask_names):
        img = cv2.imread(os.path.join(path.absolute(), image_name))
        mask = cv2.imread(os.path.join(path.absolute(), mask_name))

        img = np.where(mask, img, 0)

        # show image
        cv2.imshow("img", img)
        cv2.waitKey()


if __name__ == "__main__":
    main()
