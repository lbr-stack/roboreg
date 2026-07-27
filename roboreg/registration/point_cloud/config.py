from dataclasses import dataclass, field


@dataclass(frozen=True)
class PointCloudConfig:
    z_min: float = 0.01
    z_max: float = 2.0

    depth_conversion_factor: float = 1.0

    use_mask_boundary: bool = True
    dilation_kernel_size: int = 3
    erosion_kernel_size: int = 10


@dataclass(frozen=True)
class HydraConfig:
    reference_points_per_mesh: int = 5000

    observation: PointCloudConfig = field(default_factory=PointCloudConfig)

    max_correspondence_distance: float = 0.1
    rmse_change_tolerance: float = 1e-6  ## convergence ...


@dataclass(frozen=True)
class HydraICPConfig:
    hydra: HydraConfig = field(default_factory=HydraConfig)
    max_iterations: int = 100


@dataclass(frozen=True)
class HydraRobustICPConfig:
    hydra: HydraConfig = field(default_factory=HydraConfig)
    outer_max_iterations: int = 50
    inner_max_iterations: int = 10
