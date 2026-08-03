from dataclasses import dataclass

import numpy as np

from roboreg.core.robot import RobotData
from roboreg.registration._validation import (
    validate_extrinsics,
    validate_images,
    validate_intrinsics,
    validate_targets,
)


@dataclass(frozen=True)
class CameraData:
    intrinsics: np.ndarray
    reference_to_camera: np.ndarray | None = None

    def __post_init__(self) -> None:
        validate_intrinsics(self.intrinsics)
        if self.reference_to_camera is not None:
            validate_extrinsics(self.reference_to_camera)


@dataclass(frozen=True)
class CameraObservations:
    targets: list[np.ndarray]
    images: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("Expected at least one target.")

        if self.images is not None and len(self.images) != len(self.targets):
            raise ValueError(
                "Expected the same number of images and targets, "
                f"got {len(self.images)} and {len(self.targets)}."
            )

        validate_targets(self.targets, "targets")

        target_shape = self.targets[0].shape[:2]
        if any(target.shape[:2] != target_shape for target in self.targets):
            raise ValueError("Expected all targets to have the same shape.")

        if self.images is not None:
            validate_images(self.images, "images")

            image_shape = self.images[0].shape[:2]
            if any(image.shape[:2] != image_shape for image in self.images):
                raise ValueError("Expected all images to have the same shape.")

            if image_shape != target_shape:
                raise ValueError(
                    f"Image shape {image_shape} does not match "
                    f"target shape {target_shape}."
                )

    @property
    def shape(self) -> tuple[int, int]:
        return self.targets[0].shape[:2]


@dataclass(frozen=True)
class ImageObservations:
    joint_states: list[np.ndarray]
    cameras: dict[str, CameraObservations]


@dataclass(frozen=True)
class ImageRegistrationRequest:
    cameras: dict[str, CameraData]
    robot_data: RobotData
    observations: ImageObservations
    initial_extrinsics: np.ndarray

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("Expected at least one camera.")

        validate_extrinsics(self.initial_extrinsics)
