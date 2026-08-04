from pathlib import Path

import cv2
import numpy as np
import typer
from rich import progress

from roboreg.detector import OpenCVDetector
from roboreg.io import find_files
from roboreg.segmentor import Sam2Segmentor
from roboreg.util import overlay_mask

app = typer.Typer(add_completion=False)


@app.command()
def main(
    path: Path = typer.Option(..., help="Path to the images."),
    pattern: str = typer.Option("image_*.png", help="Image file pattern."),
    n_positive_samples: int = typer.Option(5, help="Number of positive samples."),
    n_negative_samples: int = typer.Option(5, help="Number of negative samples."),
    model_id: str = typer.Option(
        "facebook/sam2-hiera-large", help="Hugging Face model ID."
    ),
    device: str = typer.Option("cuda", help="Device to run the model. Default: cuda"),
    pre_annotated: bool = typer.Option(False, help="Try to read annotations."),
) -> None:
    r"""Generate robot masks with SAM2, seeded by OpenCV-detected samples."""
    image_files = find_files(path, pattern)

    # detect
    detector = OpenCVDetector(
        n_negative_samples=n_negative_samples,
        n_positive_samples=n_positive_samples,
    )

    # segment
    segmentor = Sam2Segmentor(model_id=model_id, device=device)

    for image_file in progress.track(image_files, description="Generating masks..."):
        img = cv2.imread(image_file)
        annotations = False
        if pre_annotated:
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
    app()
