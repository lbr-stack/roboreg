import argparse

import cv2
import numpy as np
from rich import progress

from roboreg.detector import OpenCVDetector
from roboreg.io import find_files
from roboreg.segmentor import Sam2Segmentor
from roboreg.util import overlay_mask


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        "--model-id",
        type=str,
        default="facebook/sam2-hiera-large",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model. Default: cuda",
    )
    parser.add_argument(
        "--pre-annotated",
        action="store_true",
        help="Try to read annotations.",
    )
    return parser.parse_args()


def main():
    args = args_factory()
    image_files = find_files(args.path, args.pattern)

    # detect
    detector = OpenCVDetector(
        n_negative_samples=args.n_negative_samples,
        n_positive_samples=args.n_positive_samples,
    )

    # segment
    segmentor = Sam2Segmentor(model_id=args.model_id, device=args.device)

    for image_file in progress.track(image_files, description="Generating masks..."):
        img = cv2.imread(image_file)
        annotations = False
        if args.pre_annotated:
            try:
                samples, labels = detector.read(
                    path=image_file.parent / f"{image_file.stem}_samples.csv"
                )
                annotations = True
            except FileNotFoundError:
                pass
        if not annotations:
            samples, labels = detector.detect(img)
            detector.write(
                path=image_file.parent / f"{image_file.stem}_samples.csv",
                samples=samples,
                labels=labels,
            )
        detector.clear()
        probability = segmentor(img, np.array(samples), np.array(labels))
        mask = np.where(probability > segmentor.pth, 255, 0).astype(np.uint8)
        overlay = overlay_mask(img, mask, mode="g", scale=1.0)

        # write probability and mask
        probability_path = image_file.parent / f"probability_sam2_{image_file.name}"
        mask_path = image_file.parent / f"mask_sam2_{image_file.name}"
        overlay_path = image_file.parent / f"overlay_sam2_{image_file.name}"
        cv2.imwrite(probability_path, (probability * 255.0).astype(np.uint8))
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)


if __name__ == "__main__":
    main()
