import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import rich
import rich.progress
import torch

from roboreg import differentiable as rrd
from roboreg.io import find_files, parse_camera_info
from roboreg.losses import soft_dice_loss
from roboreg.optim import LinearParticleSwarm, ParticleSwarmOptimizer
from roboreg.util import (
    look_at_from_angle,
    mask_exponential_distance_transform,
    overlay_mask,
    random_fov_eye_space_coordinates,
)


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-cameras",
        type=int,
        default=50,
        help="The number of cameras / particles to optimize.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.5,
        help="The minimum distance of the camera from the object.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=2.0,
        help="The maximum distance of the camera from the object.",
    )
    parser.add_argument(
        "--angle-range",
        type=float,
        default=np.pi,
        help="The initial angle range for the camera in [-angle_range/2, angle_range/2].",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="The maximum number of iterations.",
    )
    parser.add_argument(
        "--min-fitness-change",
        type=float,
        default=2.0e-3,
        help="The minimum fitness change for early convergence.",
    )
    parser.add_argument(
        "--max-iterations-below-min-fitness-change",
        type=int,
        default=20,
        help="The maximum number of iterations below the minimum fitness change before early convergence.",
    )
    parser.add_argument(
        "--display-progress",
        action="store_true",
        help="Display optimization progress.",
    )
    parser.add_argument(
        "--ros-package",
        type=str,
        default="lbr_description",
        help="Package where the URDF is located.",
    )
    parser.add_argument(
        "--xacro-path",
        type=str,
        default="urdf/med7/med7.xacro",
        help="Path to the xacro file, relative to --ros-package.",
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
        "--target-reduction",
        type=float,
        default=0.95,
        help="Reduces the mesh vertex count for memory reduction. In [0, 1).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Scale the camera resolution by this factor. Reduces memory usage.",
    )
    parser.add_argument(
        "--visual-meshes",
        action="store_true",
        help="If set, visual meshes will be used instead of collision meshes.",
    )
    parser.add_argument(
        "--camera-info-file",
        type=str,
        required=True,
        help="Path to the camera parameters, <path_to>/camera_info.yaml.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--image-pattern",
        type=str,
        default="image_*.png",
        help="Image file pattern. The images are only used to --display-progress.",
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
        default="image_*_mask.png",
        help="Mask file pattern.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="HT_cam_swarm.npy",
        help="Output file name. Relative to --path.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples to randomly select from the data for optimization.",
    )
    return parser.parse_args()


def parse_data(
    path: str,
    image_pattern: str,
    mask_pattern: str,
    joint_states_pattern: str,
    n_samples: int = 5,
    device: str = "cuda",
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    image_files = find_files(path, image_pattern)
    mask_files = find_files(path, mask_pattern)
    joint_states_files = find_files(path, joint_states_pattern)

    rich.print("Found the following files:")
    rich.print(f"Images: {image_files}")
    rich.print(f"Masks: {mask_files}")
    rich.print(f"Joint states: {joint_states_files}")

    # randomly sample n_samples
    if n_samples > len(image_files):
        n_samples = len(image_files)
    random_indices = np.random.choice(len(image_files), n_samples, replace=False)
    image_files = np.array(image_files)[random_indices].tolist()
    mask_files = np.array(mask_files)[random_indices].tolist()
    joint_states_files = np.array(joint_states_files)[random_indices].tolist()

    rich.print(f"Randomly sampled the following {n_samples} files:")
    rich.print(f"Images: {image_files}")
    rich.print(f"Masks: {mask_files}")
    rich.print(f"Joint states: {joint_states_files}")

    if len(mask_files) != len(joint_states_files):
        raise ValueError("Number of masks and joint states do not match.")

    images = [
        cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR) for file in image_files
    ]
    masks = [
        mask_exponential_distance_transform(
            cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        )
        for file in mask_files
    ]
    joint_states = [np.load(os.path.join(path, file)) for file in joint_states_files]

    masks = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
    joint_states = torch.tensor(
        np.array(joint_states), dtype=torch.float32, device=device
    )

    return images, joint_states, masks


def instantiate_particles(
    n_particles: int,
    height: int,
    width: int,
    focal_length_x: float,
    focal_length_y: float,
    eye_min_dist: float,
    eye_max_dist: float,
    angle_interval: float,
    device: torch.device = torch.device("cuda"),
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
        device (torch.device): The device to instantiate the particles on.

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


def main() -> None:
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    height, width, intrinsics = parse_camera_info(
        camera_info_file=args.camera_info_file
    )
    images, joint_states, masks = parse_data(
        path=args.path,
        image_pattern=args.image_pattern,
        mask_pattern=args.mask_pattern,
        joint_states_pattern=args.joint_states_pattern,
        n_samples=args.n_samples,
        device=device,
    )
    n_joint_states = joint_states.shape[0]

    # scale image data (memory reduction)
    height = int(height * args.scale)
    width = int(width * args.scale)
    intrinsics = intrinsics * args.scale
    masks = torch.nn.functional.interpolate(
        masks.unsqueeze(1), size=(height, width), mode="nearest"
    ).squeeze(1)

    # prepare particles
    particles = instantiate_particles(
        n_particles=args.n_cameras,
        height=height,
        width=width,
        focal_length_x=intrinsics[0, 0],
        focal_length_y=intrinsics[1, 1],
        eye_min_dist=args.min_distance,
        eye_max_dist=args.max_distance,
        angle_interval=args.angle_range,
        device=device,
    )
    particle_swarm = LinearParticleSwarm(
        particles=particles,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
    )

    # instantiate scene for fitness evaluation
    batch_size = (
        n_joint_states * args.n_cameras
    )  # (each camera observes n_joint_states joint states)
    camera_name = "camera"
    camera = rrd.VirtualCamera(
        resolution=(height, width),
        intrinsics=intrinsics,
        extrinsics=torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1),
        device=device,
    )

    urdf_parser = rrd.URDFParser()
    urdf_parser.from_ros_xacro(ros_package=args.ros_package, xacro_path=args.xacro_path)
    robot = rrd.Robot(
        urdf_parser=urdf_parser,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
        visual=args.visual_meshes,
        batch_size=batch_size,
        device=device,
        target_reduction=args.target_reduction,  # reduce mesh vertex count for memory reduction
    )

    renderer = rrd.NVDiffRastRenderer(device=device)
    scene = rrd.RobotScene(
        cameras={camera_name: camera},
        robot=robot,
        renderer=renderer,
    )

    # repeat joint states and masks for each camera
    masks = masks.repeat(args.n_cameras, 1, 1)
    joint_states = joint_states.repeat(args.n_cameras, 1)
    if joint_states.shape[0] != batch_size:
        raise ValueError("Joint states of invalid shape.")
    scene.robot.configure(joint_states)

    def fitness_closure() -> torch.Tensor:
        eye = particle_swarm_optimizer.particle_swarm.particles[:, :3]
        center = particle_swarm_optimizer.particle_swarm.particles[:, 3:6]
        angle = particle_swarm_optimizer.particle_swarm.particles[:, -1:]
        extrinsics = look_at_from_angle(eye=eye, center=center, angle=angle)
        scene.cameras["camera"].extrinsics = extrinsics.repeat_interleave(
            n_joint_states, 0
        )
        renders = scene.observe_from("camera").squeeze()
        fitness = (
            soft_dice_loss(renders.unsqueeze(-1), masks.unsqueeze(-1))
            .view(args.n_cameras, n_joint_states)
            .mean(dim=1)
        )
        # show the best particle of the current iteration
        if args.display_progress:
            offset = 0
            current_best_idx = torch.argmin(fitness)
            current_best_render = (
                renders[current_best_idx * n_joint_states + offset].cpu().numpy()
                * 255.0
            ).astype(np.uint8)
            # upscale render
            current_best_render = cv2.resize(
                current_best_render, (images[offset].shape[1], images[offset].shape[0])
            )
            overlay = overlay_mask(
                images[offset],
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
        max_iterations=args.max_iterations,
        min_fitness_change=args.min_fitness_change,
        max_iterations_below_min_fitness_change=args.max_iterations_below_min_fitness_change,
    )

    # save results
    best_eye = best_particle[:3].unsqueeze(0)
    best_center = best_particle[3:6].unsqueeze(0)
    best_angle = best_particle[-1:].unsqueeze(0)
    HT_cam_swarm = look_at_from_angle(
        eye=best_eye, center=best_center, angle=best_angle
    )
    np.save(os.path.join(args.path, args.output_file), HT_cam_swarm.cpu().numpy())


if __name__ == "__main__":
    main()
