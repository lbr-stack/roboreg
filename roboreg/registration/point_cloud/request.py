from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from roboreg.core.robot import RobotData
from roboreg.registration._validation import (
    validate_intrinsics,
    validate_masks,
    validate_targets,
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
        validate_targets(self.depths, "depths")

        for i, (mask, depth) in enumerate(zip(self.masks, self.depths)):
            if mask.shape != depth.shape:
                raise ValueError(
                    f"masks[{i}] and depths[{i}] have incompatible shapes: "
                    f"{mask.shape} and {depth.shape}."
                )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.depths[0].shape


@dataclass(frozen=True)
class HydraRequest:
    intrinsics: np.ndarray
    robot_data: RobotData
    observations: HydraObservations

    def __post_init__(self) -> None:
        validate_intrinsics(self.intrinsics)
