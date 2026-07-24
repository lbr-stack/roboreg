from dataclasses import dataclass
from typing import List

import numpy as np

from roboreg.core.structs import RobotData
from roboreg.reg._validation import (
    validate_depths,
    validate_extrinsics,
    validate_intrinsics,
    validate_masks,
)


@dataclass(frozen=True)
class HydraObservations:
    joint_states: List[np.ndarray]
    masks: List[np.ndarray]
    depths: List[np.ndarray]

    def __post_init__(self) -> None:
        lengths = {
            "joint_states": len(self.joint_states),
            "masks": len(self.masks),
            "depths": len(self.depths),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"All observation fields must have the same length, got {lengths}."
            )

        if not self.joint_states:
            raise ValueError("Expected at least one observation.")

        validate_masks(self.masks, "masks")
        validate_depths(self.depths, "depths")


@dataclass(frozen=True)
class HydraRequest:
    intrinsics: np.ndarray
    robot_data: RobotData
    observations: HydraObservations
    initial_extrinsics: np.ndarray

    def __post_init__(self) -> None:
        validate_intrinsics(self.intrinsics)
        validate_extrinsics(self.initial_extrinsics)
