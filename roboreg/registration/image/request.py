from dataclasses import dataclass
from typing import List

import numpy as np

from roboreg.core.robot import RobotData
from roboreg.registration._validation import (
    validate_extrinsics,
    validate_images,
    validate_intrinsics,
    validate_targets,
)


@dataclass(frozen=True)
class MonocularObservations:
    images: List[np.ndarray]
    joint_states: List[np.ndarray]
    targets: List[np.ndarray]

    def __post_init__(self) -> None:
        lengths = {
            "images": len(self.images),
            "joint_states": len(self.joint_states),
            "targets": len(self.targets),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"All observation fields must have the same length, got {lengths}."
            )

        if not self.images:
            raise ValueError("Expected at least one observation.")

        validate_images(self.images, "images")
        validate_targets(self.targets, "targets")


@dataclass(frozen=True)
class StereoObservations:
    left_images: List[np.ndarray]
    right_images: List[np.ndarray]
    joint_states: List[np.ndarray]
    left_targets: List[np.ndarray]
    right_targets: List[np.ndarray]

    def __post_init__(self) -> None:
        lengths = {
            "left_images": len(self.left_images),
            "right_images": len(self.right_images),
            "joint_states": len(self.joint_states),
            "left_targets": len(self.left_targets),
            "right_targets": len(self.right_targets),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"All observation fields must have the same length, got {lengths}."
            )

        if not self.left_images:
            raise ValueError("Expected at least one observation.")

        validate_images(self.left_images, "left_images")
        validate_images(self.right_images, "right_images")
        validate_targets(self.left_targets, "left_targets")
        validate_targets(self.right_targets, "right_targets")


@dataclass(frozen=True)
class MonocularRequest:
    intrinsics: np.ndarray
    robot_data: RobotData
    observations: MonocularObservations
    initial_extrinsics: np.ndarray

    def __post_init__(self) -> None:
        validate_intrinsics(self.intrinsics)
        validate_extrinsics(self.initial_extrinsics)


@dataclass(frozen=True)
class StereoRequest:
    left_intrinsics: np.ndarray
    right_intrinsics: np.ndarray
    initial_left_extrinsics: np.ndarray
    left_to_right_extrinsics: np.ndarray
    robot_data: RobotData
    observations: StereoObservations

    def __post_init__(self) -> None:
        validate_intrinsics(self.left_intrinsics)
        validate_intrinsics(self.right_intrinsics)
        validate_extrinsics(self.initial_left_extrinsics)
        validate_extrinsics(self.left_to_right_extrinsics)
