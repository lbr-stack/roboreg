import argparse
import os

import numpy as np
import torch

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


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--camera-info-file",
        type=str,
        required=True,
        help="Path to the camera parameters, <path_to>/camera_info.yaml.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="image_*_mask.png",
        help="Mask file pattern.",
    )
    parser.add_argument(
        "--depth-pattern",
        type=str,
        default="depth_*.npy",
        help="Depth file pattern. Note that depth values are expected in meters.",
    )
    parser.add_argument(
        "--joint-states-pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf",
        help="Path to URDF file. Meshes resolved relative to this file. "
        "Mutually exclusive with --ros-package/--xacro-path.",
    )
    parser.add_argument(
        "--ros-package",
        type=str,
        default=None,
        help="ROS package containing robot description. "
        "Requires --xacro-path. Mutually exclusive with --urdf-path.",
    )
    parser.add_argument(
        "--xacro-path",
        type=str,
        default=None,
        help="Path to xacro file relative to --ros-package. "
        "Requires --ros-package. Mutually exclusive with --urdf-path.",
    )
    parser.add_argument(
        "--root-link-name",
        type=str,
        default="",
        help="Root link name. If unspecified, the first link with mesh will be used, which may cause errors.",
    )
    parser.add_argument(
        "--end-link-name",
        type=str,
        default="",
        help="End link name. If unspecified, the last link with mesh will be used, which may cause errors.",
    )
    parser.add_argument(
        "--collision-meshes",
        action="store_true",
        help="If set, collision meshes will be used instead of visual meshes.",
    )
    parser.add_argument(
        "--depth-conversion-factor",
        type=float,
        default=1.0,
        help="Conversion factor for depth. Computes z = depth / conversion_factor e.g. to covert from millimeter to meter.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.01,
        help="Minimum depth value.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=2.0,
        help="Maximum depth value.",
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=5000,
        help="Number of points to sample from robot mesh.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.1,
        help="Maximum distance between two points to be considered as a correspondence.",
    )
    parser.add_argument(
        "--outer-max-iter",
        type=int,
        default=50,
        help="Maximum number of outer iterations.",
    )
    parser.add_argument(
        "--inner-max-iter",
        type=int,
        default=10,
        help="Maximum number of inner iterations.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="HT_hydra_robust.npy",
        help="Output file name. Relative to the path.",
    )
    parser.add_argument(
        "--no-boundary",
        action="store_true",
        help="Do not apply dilation / erosion to the mask.",
    )
    parser.add_argument(
        "--dilation-kernel-size",
        type=int,
        default=3,
        help="Dilation kernel size for mask boundary. Larger value will result in larger boundary.",
    )
    parser.add_argument(
        "--erosion-kernel-size",
        type=int,
        default=10,
        help="Erosion kernel size for mask boundary. Larger value will result in larger boundary. The closer the robot, the larger the recommended kernel size.",
    )
    parser.add_argument(
        "--display-results",
        action="store_true",
        help="Display point cloud registration results.",
    )
    validate_urdf_source(parser, parser.parse_args())
    return parser.parse_args()


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


def main():
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    joint_states_files = find_files(args.path, args.joint_states_pattern)
    mask_files = find_files(args.path, args.mask_pattern)
    depth_files = find_files(args.path, args.depth_pattern)
    observations = parse_hydra_observations(
        joint_states_files=joint_states_files,
        mask_files=mask_files,
        depth_files=depth_files,
    )
    _, _, intrinsics = parse_camera_info(args.camera_info_file)

    # instantiate robot
    if args.urdf_path is not None:
        robot_data = load_robot_data_from_urdf_file(
            urdf_path=args.urdf_path,
            root_link_name=args.root_link_name,
            end_link_name=args.end_link_name,
            collision=args.collision_meshes,
        )
    else:
        robot_data = load_robot_data_from_ros_xacro(
            ros_package=args.ros_package,
            xacro_path=args.xacro_path,
            root_link_name=args.root_link_name,
            end_link_name=args.end_link_name,
            collision=args.collision_meshes,
        )

    # register
    config = HydraRobustICPConfig(
        HydraConfig(
            reference_points_per_mesh=args.number_of_points,
            depth_to_point_cloud=DepthToPointCloudConfig(
                z_min=args.z_min,
                z_max=args.z_max,
                depth_conversion_factor=args.depth_conversion_factor,
                use_mask_boundary=not args.no_boundary,
                dilation_kernel_size=args.dilation_kernel_size,
                erosion_kernel_size=args.erosion_kernel_size,
            ),
            max_correspondence_distance=args.max_distance,
        )
    )
    hydra_robust_icp = HydraRobustICP(
        config=config,
        device=device,
        callback=visualize_hydra_result if args.display_results else None,
    )
    result = hydra_robust_icp(
        request=HydraRequest(
            intrinsics=intrinsics,
            robot_data=robot_data,
            observations=observations,
        )
    )

    # to numpy
    np.save(os.path.join(args.path, args.output_file), result.extrinsics.cpu().numpy())


if __name__ == "__main__":
    main()
