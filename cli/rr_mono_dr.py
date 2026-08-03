import argparse
import os
from pathlib import Path

import numpy as np
import torch

from roboreg.io import (
    find_files,
    load_robot_data_from_ros_xacro,
    load_robot_data_from_urdf_file,
    parse_camera_info,
    parse_monocular_observations,
)
from roboreg.registration.image.config import (
    CameraConfig,
    ConvergenceConfig,
    DiffRenderingRegistrationConfig,
    PlateauSchedulerConfig,
)
from roboreg.registration.image.objectives import (
    RenderingObjectiveType,
    create_rendering_objective,
)
from roboreg.registration.image.request import CameraData, ImageRegistrationRequest
from roboreg.registration.image.solver import DiffRenderingRegistration

from .util.validate import validate_urdf_source


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DiffRenderingRegistrationConfig().optimizer,
        help="Optimizer to use, e.g. 'Adam' or 'SGD'. Imported from torch.optim.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DiffRenderingRegistrationConfig().lr,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=ConvergenceConfig().max_iterations,
        help="Maximum number of epochs to optimize for.",
    )
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=ConvergenceConfig().tolerance,
    )
    parser.add_argument(
        "--convergence-patience",
        type=int,
        default=ConvergenceConfig().patience,
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=PlateauSchedulerConfig().factor,
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=PlateauSchedulerConfig().patience,
    )
    parser.add_argument(
        "--scheduler-threshold",
        type=float,
        default=PlateauSchedulerConfig().threshold,
    )
    parser.add_argument(
        "--rendering-objective",
        type=RenderingObjectiveType,
        choices=list(RenderingObjectiveType),
        default=RenderingObjectiveType.DISTANCE_MAP,
        help="Rendering objective.",
    )
    parser.add_argument(
        "--display-progress",
        action="store_true",
        help="Display optimization progress.",
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
        "--camera-info-file",
        type=str,
        required=True,
        help="Full path to left camera parameters, <path_to>/left_camera_info.yaml.",
    )
    parser.add_argument(
        "--extrinsics-file",
        type=str,
        required=True,
        help="Full path to homogeneous transforms from base to left camera frame, <path_to>/HT_hydra_robust.npy.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--image-pattern",
        type=str,
        default="left_image_*.png",
        help="Left image file pattern.",
    )
    parser.add_argument(
        "--joint-states-pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="left_mask_*.png",
        help="Left mask file pattern.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="HT_left_dr.npy",
        help="Left output file name. Relative to --path.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=2,
        help="Number of concurrent compilation jobs for nvdiffrast. Only relevant on first run.",
    )
    validate_urdf_source(parser, parser.parse_args())
    return parser.parse_args()


def main() -> None:
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(args.max_jobs)  # limit number of concurrent jobs
    path = Path(args.path)

    # load data
    observations = parse_monocular_observations(
        image_files=find_files(path, args.image_pattern),
        joint_states_files=find_files(path, args.joint_states_pattern),
        target_files=find_files(path, args.mask_pattern),
    )
    _, _, intrinsics = parse_camera_info(args.camera_info_file)
    extrinsics = np.load(args.extrinsics_file)

    # load robot specifications
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
    diff_rendering_registration = DiffRenderingRegistration(
        config=DiffRenderingRegistrationConfig(
            camera=CameraConfig(),
            optimizer=args.optimizer,
            lr=args.lr,
            convergence=ConvergenceConfig(
                max_iterations=args.max_iterations,
                tolerance=args.convergence_tolerance,
                patience=args.convergence_patience,
            ),
            plateau_scheduler=PlateauSchedulerConfig(
                mode="min",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                threshold=args.scheduler_threshold,
            ),
        ),
        objective=create_rendering_objective(objective_type=args.rendering_objective),
        device=device,
    )
    result = diff_rendering_registration(
        request=ImageRegistrationRequest(
            cameras={
                "camera": CameraData(
                    intrinsics=intrinsics,
                )
            },
            robot_data=robot_data,
            observations=observations,
            initial_extrinsics=extrinsics,
        )
    )

    # save extrinsics
    np.save(
        path / args.output_file,
        result.extrinsics.cpu().numpy(),
    )


if __name__ == "__main__":
    main()
