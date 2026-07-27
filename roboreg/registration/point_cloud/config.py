from dataclasses import dataclass, field


@dataclass(frozen=True)
class DepthToPointCloudConfig:
    z_min: float = 0.01
    z_max: float = 2.0
    depth_conversion_factor: float = 1.0

    use_mask_boundary: bool = True
    dilation_kernel_size: int = 3
    erosion_kernel_size: int = 10


@dataclass(frozen=True)
class HydraConfig:
    reference_points_per_mesh: int = 5000

    depth_to_point_cloud: DepthToPointCloudConfig = field(
        default_factory=DepthToPointCloudConfig
    )

    max_correspondence_distance: float = 0.1
    rmse_change_tolerance: float = 1e-6


@dataclass(frozen=True)
class HydraICPConfig:
    hydra: HydraConfig = field(default_factory=HydraConfig)
    max_iterations: int = 100


@dataclass(frozen=True)
class HydraRobustICPConfig:
    hydra: HydraConfig = field(default_factory=HydraConfig)
    max_outer_iterations: int = 50
    max_inner_iterations: int = 10
