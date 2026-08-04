import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import typer

from roboreg.core import NVDiffRastRenderer, Robot, RobotScene, VirtualCamera
from roboreg.io import (
    find_files,
    load_robot_data_from_ros_xacro,
    load_robot_data_from_urdf_file,
    parse_camera_info,
    parse_monocular_observations,
)
from roboreg.losses import soft_dice_loss
from roboreg.optim import LinearParticleSwarm, ParticleSwarmOptimizer
from roboreg.util import (
    look_at_from_angle,
    mask_exponential_decay,
    overlay_mask,
    random_fov_eye_space_coordinates,
)

from .util.validate import validate_urdf_source

app = typer.Typer(add_completion=False)


def instantiate_particles(
    n_particles: int,
    height: int,
    width: int,
    focal_length_x: float,
    focal_length_y: float,
    eye_min_dist: float,
    eye_max_dist: float,
    angle_interval: float,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    r"""Instantiate the particles for the optimization randomly under field of view constraints.
    Particles (camera poses) are represented using eye space coordinates (eye, center, angle).

    Args:
        n_particles (int): The number of particles to instantiate.
        height (int): The height of the image.
        width (int): The width of the image.
        focal_length_x (float): The focal length in x direction.
        focal_length_y (float): The focal length in y direction.
        eye_min_dist (float): The minimum distance of the eye from the origin.
        eye_max_dist (float): The maximum distance of the eye from the origin.
        angle_interval (float): The angle interval in which to sample the rotation angle.
        device (Union[torch.device, str]): The device to instantiate the particles on.

    Returns:
        torch.Tensor: The particles of shape (n_particles, 7).
    """
    heights = torch.full([n_particles], height, dtype=torch.float32, device=device)
    widths = torch.full([n_particles], width, dtype=torch.float32, device=device)
    focal_lengths_x = torch.full(
        [n_particles], focal_length_x, dtype=torch.float32, device=device
    )
    focal_lengths_y = torch.full(
        [n_particles], focal_length_y, dtype=torch.float32, device=device
    )
    eye_min_dists = torch.full(
        [n_particles], eye_min_dist, dtype=torch.float32, device=device
    )
    eye_max_dists = torch.full(
        [n_particles], eye_max_dist, dtype=torch.float32, device=device
    )
    angle_intervals = torch.full(
        [n_particles], angle_interval, dtype=torch.float32, device=device
    )

    random_eyes, random_centers, random_angles = random_fov_eye_space_coordinates(
        heights=heights,
        widths=widths,
        focal_lengths_x=focal_lengths_x,
        focal_lengths_y=focal_lengths_y,
        eye_min_dists=eye_min_dists,
        eye_max_dists=eye_max_dists,
        angle_intervals=angle_intervals,
    )

    return torch.cat([random_eyes, random_centers, random_angles], dim=-1)


@app.command()
def main(
    camera_info_file: Path = typer.Option(
        ..., help="Path to the camera parameters, <path_to>/camera_info.yaml."
    ),
    path: Path = typer.Option(..., help="Path to the data."),
    n_cameras: int = typer.Option(
        50, help="The number of cameras / particles to optimize."
    ),
    min_distance: float = typer.Option(
        0.5, help="The minimum distance of the camera from the object."
    ),
    max_distance: float = typer.Option(
        2.0, help="The maximum distance of the camera from the object."
    ),
    angle_range: float = typer.Option(
        np.pi,
        help="The initial angle range for the camera in [-angle_range/2, angle_range/2].",
    ),
    w: float = typer.Option(0.7),
    c1: float = typer.Option(1.5),
    c2: float = typer.Option(1.5),
    max_iterations: int = typer.Option(100, help="The maximum number of iterations."),
    min_fitness_change: float = typer.Option(
        2.0e-3, help="The minimum fitness change for early convergence."
    ),
    max_iterations_below_min_fitness_change: int = typer.Option(
        20,
        help="The maximum number of iterations below the minimum fitness change before early convergence.",
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
    target_reduction: float = typer.Option(
        0.95,
        help="Reduces the mesh vertex count for memory reduction. In [0, 1).",
    ),
    scale: float = typer.Option(
        0.25, help="Scale the camera resolution by this factor. Reduces memory usage."
    ),
    collision_meshes: bool = typer.Option(
        False, help="If set, collision meshes will be used instead of visual meshes."
    ),
    image_pattern: str = typer.Option(
        "image_*.png",
        help="Image file pattern. The images are only used to --display-progress.",
    ),
    joint_states_pattern: str = typer.Option(
        "joint_states_*.npy", help="Joint state file pattern."
    ),
    mask_pattern: str = typer.Option("image_*_mask.png", help="Mask file pattern."),
    output_file: str = typer.Option(
        "HT_cam_swarm.npy", help="Output file name. Relative to --path."
    ),
    n_samples: int = typer.Option(
        5,
        help="Number of samples to randomly select from the data for optimization.",
    ),
    max_jobs: int = typer.Option(
        2,
        help="Number of concurrent compilation jobs for nvdiffrast. Only relevant on first run.",
    ),
) -> None:
    r"""Particle swarm optimization for an initial camera pose estimate."""
    validate_urdf_source(urdf_path, ros_package, xacro_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(max_jobs)  # limit number of concurrent jobs

    # load data
    height, width, intrinsics = parse_camera_info(camera_info_file=camera_info_file)
    image_files = find_files(path, image_pattern)
    target_files = find_files(path, mask_pattern)
    joint_states_files = find_files(path, joint_states_pattern)
    if n_samples > len(image_files):  # randomly sample n_samples
        n_samples = len(image_files)
    random_indices = np.random.choice(len(image_files), n_samples, replace=False)
    image_files = np.array(image_files)[random_indices].tolist()
    target_files = np.array(target_files)[random_indices].tolist()
    joint_states_files = np.array(joint_states_files)[random_indices].tolist()
    observations = parse_monocular_observations(
        image_files=image_files,
        target_files=target_files,
        joint_states_files=joint_states_files,
    )

    # pre-process data
    camera_name = "camera"
    joint_states = torch.tensor(
        np.array(observations.joint_states), dtype=torch.float32, device=device
    )
    n_joint_states = joint_states.shape[0]
    masks = [
        mask_exponential_decay(mask)
        for mask in observations.cameras[camera_name].targets
    ]
    masks = torch.tensor(np.array(masks), dtype=torch.float32, device=device)

    # scale image data (memory reduction)
    height = int(height * scale)
    width = int(width * scale)
    intrinsics = intrinsics * scale
    masks = torch.nn.functional.interpolate(
        masks.unsqueeze(1), size=(height, width), mode="nearest"
    ).squeeze(1)

    # prepare particles
    particles = instantiate_particles(
        n_particles=n_cameras,
        height=height,
        width=width,
        focal_length_x=intrinsics[0, 0],
        focal_length_y=intrinsics[1, 1],
        eye_min_dist=min_distance,
        eye_max_dist=max_distance,
        angle_interval=angle_range,
        device=device,
    )
    particle_swarm = LinearParticleSwarm(
        particles=particles,
        w=w,
        c1=c1,
        c2=c2,
    )

    # instantiate scene for fitness evaluation
    batch_size = (
        n_joint_states * n_cameras
    )  # (each camera observes n_joint_states joint states)
    camera = VirtualCamera(
        resolution=(height, width),
        intrinsics=intrinsics,
        extrinsics=torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1),
        device=device,
    )

    # instantiate robot
    if urdf_path is not None:
        robot_data = load_robot_data_from_urdf_file(
            urdf_path=urdf_path,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision_meshes,
            target_reduction=target_reduction,
        )
    else:
        robot_data = load_robot_data_from_ros_xacro(
            ros_package=ros_package,
            xacro_path=xacro_path,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision_meshes,
            target_reduction=target_reduction,
        )
    robot = Robot.from_robot_data(
        robot_data=robot_data, batch_size=batch_size, device=device
    )

    # instantiate scene
    scene = RobotScene(
        cameras={camera_name: camera},
        robot=robot,
        renderer=NVDiffRastRenderer(device=device),
    )

    # repeat joint states and masks for each camera
    masks = masks.repeat(n_cameras, 1, 1)
    joint_states = joint_states.repeat(n_cameras, 1)
    if joint_states.shape[0] != batch_size:
        raise ValueError("Joint states of invalid shape.")
    scene.robot.configure(joint_states)

    def fitness_closure() -> torch.Tensor:
        eye = particle_swarm_optimizer.particle_swarm.particles[:, :3]
        center = particle_swarm_optimizer.particle_swarm.particles[:, 3:6]
        angle = particle_swarm_optimizer.particle_swarm.particles[:, -1:]
        extrinsics = look_at_from_angle(eye=eye, center=center, angle=angle)
        scene.cameras[camera_name].extrinsics = extrinsics.repeat_interleave(
            n_joint_states, 0
        )
        renders = scene.observe_from(camera_name).squeeze()
        fitness = (
            soft_dice_loss(renders.unsqueeze(-1), masks.unsqueeze(-1))
            .view(n_cameras, n_joint_states)
            .mean(dim=1)
        )
        # show the best particle of the current iteration
        if display_progress:
            offset = 0
            current_best_idx = torch.argmin(fitness)
            current_best_render = (
                renders[current_best_idx * n_joint_states + offset].cpu().numpy()
                * 255.0
            ).astype(np.uint8)
            # upscale render
            current_best_render = cv2.resize(
                current_best_render,
                (
                    observations.cameras[camera_name].images[offset].shape[1],
                    observations.cameras[camera_name].images[offset].shape[0],
                ),
            )
            overlay = overlay_mask(
                observations.cameras[camera_name].images[offset],
                current_best_render,
                scale=1.0,
            )
            cv2.imshow("Best particle of current iteration", overlay)
            cv2.waitKey(1)
        return fitness

    # prepare optimizer
    particle_swarm_optimizer = ParticleSwarmOptimizer(
        particle_swarm=particle_swarm,
    )

    # optimize
    best_particle, _ = particle_swarm_optimizer(
        fitness_function=fitness_closure,
        max_iterations=max_iterations,
        min_fitness_change=min_fitness_change,
        max_iterations_below_min_fitness_change=max_iterations_below_min_fitness_change,
    )

    # save results
    best_eye = best_particle[:3].unsqueeze(0)
    best_center = best_particle[3:6].unsqueeze(0)
    best_angle = best_particle[-1:].unsqueeze(0)
    HT_cam_swarm = look_at_from_angle(
        eye=best_eye, center=best_center, angle=best_angle
    )
    np.save(path / output_file, HT_cam_swarm.cpu().numpy())


if __name__ == "__main__":
    app()
