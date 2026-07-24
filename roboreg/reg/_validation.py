from typing import List

import numpy as np


def validate_intrinsics(intrinsics: np.ndarray) -> None:
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must have shape (3, 3), got {intrinsics.shape}.")


def validate_extrinsics(extrinsics: np.ndarray) -> None:
    if extrinsics.shape != (4, 4):
        raise ValueError(
            "Extrinsics must have shape (4, 4), " f"got {extrinsics.shape}."
        )


def validate_images(images: List[np.ndarray], name: str) -> None:
    for index, image in enumerate(images):
        if image.ndim != 3:
            raise ValueError(f"{name}[{index}] must be 3D, got shape {image.shape}.")

        if image.shape[-1] != 3:
            raise ValueError(
                f"{name}[{index}] must have 3 channels, got shape {image.shape}."
            )


def validate_masks(masks: List[np.ndarray], name: str) -> None:
    for index, mask in enumerate(masks):
        if mask.ndim != 2:
            raise ValueError(f"{name}[{index}] must be 2D, got shape {mask.shape}.")

        if mask.dtype != np.uint8:
            raise ValueError(
                f"{name}[{index}] must have dtype np.uint8, got {mask.dtype}."
            )


def validate_depths(depths: List[np.ndarray], name: str) -> None:
    for index, depth in enumerate(depths):
        if depth.ndim != 2:
            raise ValueError(f"{name}[{index}] must be 2D, got shape {depth.shape}.")
