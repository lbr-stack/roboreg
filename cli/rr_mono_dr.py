import os
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
    parse_intrinsics,
    parse_monocular_observations,
)
from roboreg.registration.image.callbacks import RenderOverlayCallback
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
from roboreg.registration.image.solver import (
    DiffRenderingRegistration,
    OptimizationCallback,
    OptimizationState,
)

from .util.validate import validate_urdf_source

app = typer.Typer(add_completion=False)


def print_optimization_state(state: OptimizationState) -> None:
    rich.print(
        f"Step [{state.iteration} / {state.max_iterations}], "
        f"loss: {state.loss:.3f}, "
        f"best loss: {state.best_loss:.3f}, "
        f"lr: {state.learning_rate:.3e}"
    )


@app.command()
def main(
    intrinsics_file: Path = typer.Option(
        ...,
        help="Full path to intrinsics, e.g. <path_to>/intrinsics.csv or <path_to>/camera_info.yaml.",
    ),
    extrinsics_file: Path = typer.Option(
        ...,
        help="Full path to homogeneous transforms from base to left camera frame, <path_to>/HT_hydra_robust.npy.",
    ),
    path: Path = typer.Option(..., help="Path to the data."),
    optimizer: str = typer.Option(
        DiffRenderingRegistrationConfig().optimizer,
        help="Optimizer to use, e.g. 'Adam' or 'SGD'. Imported from torch.optim.",
    ),
    lr: float = typer.Option(
        DiffRenderingRegistrationConfig().lr, help="Learning rate for the optimizer."
    ),
    max_iterations: int = typer.Option(
        ConvergenceConfig().max_iterations,
        help="Maximum number of epochs to optimize for.",
    ),
    convergence_tolerance: float = typer.Option(ConvergenceConfig().tolerance),
    convergence_patience: int = typer.Option(ConvergenceConfig().patience),
    scheduler_factor: float = typer.Option(PlateauSchedulerConfig().factor),
    scheduler_patience: int = typer.Option(PlateauSchedulerConfig().patience),
    scheduler_threshold: float = typer.Option(PlateauSchedulerConfig().threshold),
    rendering_objective: RenderingObjectiveType = typer.Option(
        RenderingObjectiveType.DISTANCE_MAP, help="Rendering objective."
    ),
    display_progress: bool = typer.Option(False, help="Display optimization progress."),
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
    image_pattern: str = typer.Option(
        "left_image_*.png", help="Left image file pattern."
    ),
    joint_states_pattern: str = typer.Option(
        "joint_states_*.npy", help="Joint state file pattern."
    ),
    mask_pattern: str = typer.Option("left_mask_*.png", help="Left mask file pattern."),
    output_file: str = typer.Option(
        "HT_left_dr.npy", help="Left output file name. Relative to --path."
    ),
    max_jobs: int = typer.Option(
        2,
        help="Number of concurrent compilation jobs for nvdiffrast. Only relevant on first run.",
    ),
) -> None:
    r"""Monocular differentiable rendering registration."""
    validate_urdf_source(urdf_path, ros_package, xacro_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(max_jobs)  # limit number of concurrent jobs

    # load data
    observations = parse_monocular_observations(
        image_files=find_files(path, image_pattern),
        joint_states_files=find_files(path, joint_states_pattern),
        target_files=find_files(path, mask_pattern),
    )
    intrinsics = parse_intrinsics(intrinsics_file)
    extrinsics = np.load(extrinsics_file)

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
    on_iteration: list[OptimizationCallback] = [
        print_optimization_state,
    ]
    if display_progress:
        on_iteration.append(
            RenderOverlayCallback(
                images={
                    camera_name: camera_observations.images
                    for camera_name, camera_observations in observations.cameras.items()
                    if camera_observations.images is not None
                },
            )
        )
    diff_rendering_registration = DiffRenderingRegistration(
        config=DiffRenderingRegistrationConfig(
            camera=CameraConfig(),
            optimizer=optimizer,
            lr=lr,
            convergence=ConvergenceConfig(
                max_iterations=max_iterations,
                tolerance=convergence_tolerance,
                patience=convergence_patience,
            ),
            plateau_scheduler=PlateauSchedulerConfig(
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
                threshold=scheduler_threshold,
            ),
        ),
        objective=create_rendering_objective(objective_type=rendering_objective),
        device=device,
        on_iteration=on_iteration,
    )
    rich.print("Entering optimization...")
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
    rich.print(
        f"Optimization terminated after {result.iterations} iterations "
        f"with status '{result.termination_reason}'."
    )

    # save extrinsics
    rich.print(f"Writing results to: '{path}'.")
    np.save(
        path / output_file,
        result.extrinsics.cpu().numpy(),
    )


if __name__ == "__main__":
    app()
