from pathlib import Path
from typing import Optional

import numpy as np
import rich
import torch
import typer

from roboreg.io import (
    find_files,
    load_robot_data_from_ros_xacro,
    load_robot_data_from_urdf_file,
    parse_camera_info,
    parse_hydra_observations,
)
from roboreg.registration.point_cloud.config import (
    DepthToPointCloudConfig,
    HydraConfig,
    HydraRobustICPConfig,
)
from roboreg.registration.point_cloud.request import HydraRequest
from roboreg.registration.point_cloud.solver import HydraProblem, HydraRobustICP
from roboreg.registration.result import RegistrationResult

from .util.validate import validate_urdf_source

app = typer.Typer(add_completion=False)


def visualize_hydra_result(
    problem: HydraProblem,
    result: RegistrationResult,
) -> None:
    from roboreg.util import RegistrationVisualizer

    visualizer = RegistrationVisualizer()

    visualizer(
        mesh_vertices=problem.reference_vertices,
        observed_vertices=problem.observed_vertices,
    )

    visualizer(
        mesh_vertices=problem.reference_vertices,
        observed_vertices=problem.observed_vertices,
        HT=torch.linalg.inv(result.extrinsics),
    )


@app.command()
def main(
    camera_info_file: Path = typer.Option(
        ..., help="Path to the camera parameters, <path_to>/camera_info.yaml."
    ),
    path: Path = typer.Option(..., help="Path to the data."),
    mask_pattern: str = typer.Option("image_*_mask.png", help="Mask file pattern."),
    depth_pattern: str = typer.Option(
        "depth_*.npy",
        help="Depth file pattern. Note that depth values are expected in meters.",
    ),
    joint_states_pattern: str = typer.Option(
        "joint_states_*.npy", help="Joint state file pattern."
    ),
    urdf_path: Optional[Path] = typer.Option(
        "test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf",
        help="Path to URDF file. Meshes resolved relative to this file. "
        "Mutually exclusive with --ros-package/--xacro-path.",
    ),
    ros_package: Optional[str] = typer.Option(
        None,
        help="ROS package containing robot description. "
        "Requires --xacro-path. Mutually exclusive with --urdf-path.",
    ),
    xacro_path: Optional[str] = typer.Option(
        None,
        help="Path to xacro file relative to --ros-package. "
        "Requires --ros-package. Mutually exclusive with --urdf-path.",
    ),
    root_link_name: str = typer.Option(
        "",
        help="Root link name. If unspecified, the first link with mesh will be used, which may cause errors.",
    ),
    end_link_name: str = typer.Option(
        "",
        help="End link name. If unspecified, the last link with mesh will be used, which may cause errors.",
    ),
    collision_meshes: bool = typer.Option(
        False, help="If set, collision meshes will be used instead of visual meshes."
    ),
    depth_conversion_factor: float = typer.Option(
        1.0,
        help="Conversion factor for depth. Computes z = depth / conversion_factor e.g. to covert from millimeter to meter.",
    ),
    z_min: float = typer.Option(0.01, help="Minimum depth value."),
    z_max: float = typer.Option(2.0, help="Maximum depth value."),
    number_of_points: int = typer.Option(
        5000, help="Number of points to sample from robot mesh."
    ),
    max_distance: float = typer.Option(
        0.1,
        help="Maximum distance between two points to be considered as a correspondence.",
    ),
    max_outer_iterations: int = typer.Option(
        50, help="Maximum number of outer iterations."
    ),
    max_inner_iterations: int = typer.Option(
        10, help="Maximum number of inner iterations."
    ),
    output_file: str = typer.Option(
        "HT_hydra_robust.npy", help="Output file name. Relative to the path."
    ),
    no_boundary: bool = typer.Option(
        False, help="Do not apply dilation / erosion to the mask."
    ),
    dilation_kernel_size: int = typer.Option(
        3,
        help="Dilation kernel size for mask boundary. Larger value will result in larger boundary.",
    ),
    erosion_kernel_size: int = typer.Option(
        10,
        help="Erosion kernel size for mask boundary. Larger value will result in larger boundary. The closer the robot, the larger the recommended kernel size.",
    ),
    display_results: bool = typer.Option(
        False, help="Display point cloud registration results."
    ),
) -> None:
    r"""Hydra robust ICP: point-to-plane ICP registration on a Lie algebra."""
    validate_urdf_source(urdf_path, ros_package, xacro_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    observations = parse_hydra_observations(
        joint_states_files=find_files(path, joint_states_pattern),
        mask_files=find_files(path, mask_pattern),
        depth_files=find_files(path, depth_pattern),
    )
    _, _, intrinsics = parse_camera_info(camera_info_file)

    # load robot specifications
    if urdf_path is not None:
        robot_data = load_robot_data_from_urdf_file(
            urdf_path=urdf_path,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision_meshes,
        )
    else:
        robot_data = load_robot_data_from_ros_xacro(
            ros_package=ros_package,
            xacro_path=xacro_path,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision_meshes,
        )

    # register
    config = HydraRobustICPConfig(
        HydraConfig(
            reference_points_per_mesh=number_of_points,
            depth_to_point_cloud=DepthToPointCloudConfig(
                z_min=z_min,
                z_max=z_max,
                depth_conversion_factor=depth_conversion_factor,
                use_mask_boundary=not no_boundary,
                dilation_kernel_size=dilation_kernel_size,
                erosion_kernel_size=erosion_kernel_size,
            ),
            max_correspondence_distance=max_distance,
        ),
        max_outer_iterations=max_outer_iterations,
        max_inner_iterations=max_inner_iterations,
    )
    hydra_robust_icp = HydraRobustICP(
        config=config,
        device=device,
        on_after_registration=visualize_hydra_result if display_results else None,
    )
    rich.print("Entering optimization...")
    result = hydra_robust_icp(
        request=HydraRequest(
            intrinsics=intrinsics,
            robot_data=robot_data,
            observations=observations,
        )
    )
    rich.print(
        f"Optimization terminated after {result.iterations} iterations "
        f"with status '{result.termination_reason}'."
    )

    # save extrinsics
    rich.print(f"Writing results to: '{path}'.")
    np.save(path / output_file, result.extrinsics.cpu().numpy())


if __name__ == "__main__":
    app()
