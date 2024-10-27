import argparse
import os
import pathlib

import cv2
import numpy as np
from rich import progress

from roboreg.detector import OpenCVDetector
from roboreg.io import find_files
from roboreg.segmentor import SamSegmentor
from roboreg.util import overlay_mask


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the images.")
    parser.add_argument(
        "--pattern", type=str, default="image_*.png", help="Image file pattern."
    )
    parser.add_argument(
        "--n-positive-samples", type=int, default=5, help="Number of positive samples."
    )
    parser.add_argument(
        "--n-negative-samples", type=int, default=5, help="Number of negative samples."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Full path to SAM checkpoint. Should be named ~sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model. Default: cuda",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        help="Type of the model. Default: vit_h",
    )
    parser.add_argument(
        "--pre-annotated",
        action="store_true",
        help="Try to read annotations.",
    )
    return parser.parse_args()


def main():
    args = args_factory()
    path = pathlib.Path(args.path)
    image_names = find_files(path.absolute(), args.pattern)

    # detect
    detector = OpenCVDetector(
        n_positive_samples=args.n_positive_samples,
        n_negative_samples=args.n_negative_samples,
    )

    # segment
    segmentor = SamSegmentor(
        checkpoint=args.checkpoint, model_type=args.model_type, device=args.device
    )

    for image_name in progress.track(image_names, description="Generating masks..."):
        image_prefix = image_name.split(".")[0]
        image_suffix = image_name.split(".")[1]
        img = cv2.imread(os.path.join(path.absolute(), image_name))
        annotations = False
        if args.pre_annotated:
            try:
                samples, labels = detector.read(
                    path=os.path.join(path.absolute(), f"{image_prefix}_samples.csv")
                )
                annotations = True
            except FileNotFoundError:
                pass
        if not annotations:
            samples, labels = detector.detect(img)
            detector.write(
                path=os.path.join(path.absolute(), f"{image_prefix}_samples.csv"),
                samples=samples,
                labels=labels,
            )
        detector.clear()
        probability = segmentor(img, np.array(samples), np.array(labels))
        mask = np.where(probability > segmentor.pth, 255, 0).astype(np.uint8)
        overlay = overlay_mask(img, mask, mode="g", scale=1.0)

        # write probability and mask
        probability_path = os.path.join(
            path.absolute(), f"probability_sam_{image_prefix}.{image_suffix}"
        )
        mask_path = os.path.join(
            path.absolute(), f"mask_sam_{image_prefix}.{image_suffix}"
        )
        overlay_path = os.path.join(
            path.absolute(), f"mask_overlay_sam_{image_prefix}.{image_suffix}"
        )
        cv2.imwrite(probability_path, (probability * 255.0).astype(np.uint8))
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)


if __name__ == "__main__":
    main()
