import math
from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass(frozen=True)
class CameraConfig:
    z_min: float = 0.1
    z_max: float = 100.0
    target_resolution: Tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.z_max <= self.z_min:
            raise ValueError("z_max must be greater than z_min.")
        if self.z_min < 0:
            raise ValueError("z_min must be greater equal zero.")

        if self.target_resolution is not None:
            height, width = self.target_resolution

            if height <= 0 or width <= 0:
                raise ValueError("target_resolution dimensions must be positive.")


@dataclass(frozen=True)
class ConvergenceConfig:
    max_iterations: int = 400
    tolerance: float = 1.0e-3
    patience: int = 50

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")


@dataclass(frozen=True)
class PlateauSchedulerConfig:
    mode: Literal["min", "max"] = "min"
    factor: float = 0.1
    patience: int = 50
    threshold: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.factor <= 0 or self.factor >= 1:
            raise ValueError("factor must be in the range (0, 1).")
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative.")


@dataclass(frozen=True)
class DiffRenderingRegistrationConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)

    optimizer: str = "AdamW"
    lr: float = 3.0e-2

    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    plateau_scheduler: PlateauSchedulerConfig = field(
        default_factory=PlateauSchedulerConfig
    )

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive.")


@dataclass(frozen=True)
class CameraSwarmRegistrationConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)

    n_cameras: int = 50
    min_distance: float = 0.5
    max_distance: float = 2.0
    angle_range: float = math.pi

    inertia_weight: float = 0.7
    cognitive_coefficient: float = 1.5
    social_coefficient: float = 1.5

    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    def __post_init__(self) -> None:
        if self.n_cameras <= 0:
            raise ValueError("n_cameras must be positive.")

        if self.min_distance <= 0:
            raise ValueError("min_distance must be positive.")

        if self.max_distance <= self.min_distance:
            raise ValueError("max_distance must be greater than min_distance.")

        if self.angle_range <= 0:
            raise ValueError("angle_range must be positive.")
