import os
from abc import ABC, abstractmethod

import numpy as np
import rich
import torch

from roboreg import differentiable as rrd
from roboreg.io import find_files, parse_camera_info
from roboreg.losses import soft_dice_loss
from roboreg.util import random_fov_eye_space_coordinates

from .particle_swarm import ParticleSwarmOptimizer


class CameraSwarmOptimizer(ParticleSwarmOptimizer):
    def __init__(self) -> None:
        super().__init__()

        # kinematics, meshes, virtual camera ?????
        # references, joint states, masks, camera intrinsics, height, width, downscaling factor, device
        # to be downsampled????

        # height = 1
        # width = 1
        # focal_length_x = 1.0
        # focal_length_y = 1.0
        # eye_min_dist = 1.0
        # eye_max_dist = 5.0
        # angle_interval = torch.pi
        # batch_size = 1
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # ## add height, width, fov,....
        # random_fov_eye_space_coordinates(

        # )

    def _fitness_function(self, particles: torch.Tensor) -> torch.Tensor:
        # TODO: implement this averaging over joint configurations
        # .view(n_camera_poses, n_joint_configurations) #### how to abstract this???
        # .mean(dim=1)

        print(f"grad enabled: {torch.is_grad_enabled()}")


    def _init_random_particles(self) -> torch.Tensor:
        # initialize using random_fov....
        pass

"""
def test_particle_swarm() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_camera_poses = 100  # can also be considered particles here...
    downscaling_factor = 4
    path = "roboreg/test/data/lbr_med7/zed2i/stereo_data"
    joint_state_pattern = "joint_state_*.npy"
    mask_pattern = "left_mask_*.png"
    ros_package = "lbr_description"
    xacro_path = "urdf/med7/med7.xacro"
    camera_info_file = "left_camera_info.yaml"
    gt_ht_file = "HT_left_dr.npy"
    n_desired_joint_states = 6
    max_iter = 200
    terminate_loss = 0.05  # TODO: replace this with an epsilon
    target_reduction = 0.95
    visualize = True

    # this snipper seeks to initialize a pose estimate through a random search
    generator = RandomExtrinsicsGenerator()

    # load data
    joint_state_files = find_files(path, joint_state_pattern)
    joint_states = [
        np.load(os.path.join(path, joint_state_file))
        for joint_state_file in joint_state_files
    ]
    joint_states = torch.tensor(
        np.array(joint_states[:n_desired_joint_states]),
        dtype=torch.float32,
        device=device,
    )

    n_joint_configurations = joint_states.shape[0]
    batch_size = n_joint_configurations * n_camera_poses

    mask_files = find_files(path, mask_pattern)
    masks = [
        cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE)
        for mask_file in mask_files
    ]
    masks_tensor = (
        torch.tensor(
            np.array(masks[:n_desired_joint_states]), dtype=torch.float32, device=device
        )
        / 255.0
    )
    masks_tensor = masks_tensor.repeat(n_camera_poses, 1, 1)

    # repeat joint states for each pose
    joint_states = joint_states.repeat(n_camera_poses, 1)

    if joint_states.shape[0] != batch_size:
        raise ValueError(
            f"Expected joint_states of batch size {batch_size}, got {joint_states.shape[0]}"
        )

    # instantiate kinematics and meshes
    urdf_parser = rrd.URDFParser()
    urdf_parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)
    root_link_name = urdf_parser.link_names_with_meshes(visual=False)[0]
    end_link_name = urdf_parser.link_names_with_meshes(visual=False)[-1]

    kinematics = rrd.TorchKinematics(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )
    meshes = rrd.TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name, visual=False
        ),
        batch_size=batch_size,
        device=device,
        target_reduction=target_reduction,
    )

    # instantiate camera
    height, width, intrinsics = parse_camera_info(os.path.join(path, camera_info_file))

    # downscale 2x
    height = height // downscaling_factor
    width = width // downscaling_factor
    intrinsics = intrinsics / downscaling_factor
    masks_tensor = torch.nn.functional.interpolate(
        masks_tensor.unsqueeze(1), size=(height, width), mode="nearest"
    ).squeeze(1)

    camera = rrd.VirtualCamera(
        resolution=(height, width),
        intrinsics=intrinsics,
        extrinsics=torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1),
        device=device,
    )

    # instantiate scene
    renderer = rrd.NVDiffRastRenderer(device=device)
    scene = rrd.RobotScene(
        meshes=meshes,
        kinematics=kinematics,
        renderer=renderer,
        cameras={"camera": camera},
    )

    # configure robot joint states
    scene.configure_robot_joint_states(joint_states)

    # initialize random particle (particles as in batch dimension...)
    random_particles = generator.generate_random_variables(
        height=height,
        width=width,
        focal_length_x=intrinsics[0, 0],
        focal_length_y=intrinsics[1, 1],
        eye_min_dist=0.5,
        eye_max_dist=2.0,
        angle_interval=torch.pi,
        batch_size=n_camera_poses,
        device=device,
    )

    # track best value of this particle and best global particle
    best_particles = random_particles.clone()  # B x 7 DoF (eye, center, angle)
    best_particle = torch.zeros_like(random_particles[0])  # 7 DoF
    best_particle_losses = torch.full_like(random_particles[:, 0], float("inf"))  # B
    best_global_loss = float("inf")

    @torch.no_grad()
    def fitness_function(random_variable: torch.Tensor) -> torch.Tensor:
        extrinsics = RandomExtrinsicsGenerator.generate_view(
            random_variable=random_variable,
            batch_size=random_variable.shape[0],
            device=random_variable.device,
        )
        scene.cameras["camera"].extrinsics = extrinsics.repeat_interleave(
            n_joint_configurations, 0
        )
        renders = scene.observe_from("camera").squeeze()
        loss = soft_dice_loss(renders, masks_tensor)
        return loss

    velocity = torch.zeros_like(random_particles)

    # live 3d plot
    if visualize:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("TkAgg")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            random_particles[:, 0].cpu().numpy(),
            random_particles[:, 1].cpu().numpy(),
            random_particles[:, 2].cpu().numpy(),
        )
        plt.ion()
        plt.show()

    gt_extrinsics = np.load(os.path.join(path, gt_ht_file))
    try:
        while (
            not best_global_loss < terminate_loss
        ):  # TODO: replace with change in loss
            # compute particle veloctiy
            w = 0.7
            c1 = 1.5
            c2 = 1.5
            r1 = torch.rand_like(random_particles)
            r2 = torch.rand_like(random_particles)

            # update particle velocity
            velocity = (
                w * velocity
                + c1 * r1 * (best_particles - random_particles)
                + c2 * r2 * (best_particle - random_particles)
            )  ### how to guarantee these are within bounds?

            # update particle position
            random_particles = random_particles + velocity

            # evaluate fitness
            losses = (
                fitness_function(random_particles)
                .view(n_camera_poses, n_joint_configurations)
                .mean(dim=1)
            )

            # update particles best known losses and positions
            mask = losses < best_particle_losses
            best_particle_losses[mask] = losses[mask]
            best_particles[mask] = random_particles[mask]

            if visualize:
                sc._offsets3d = (
                    random_particles[:, 0].cpu().numpy(),
                    random_particles[:, 1].cpu().numpy(),
                    random_particles[:, 2].cpu().numpy(),
                )
                plt.draw()
                plt.pause(0.05)

            # update global best
            best_loss_idx = torch.argmin(best_particle_losses)
            best_particle_loss = best_particle_losses[best_loss_idx]
            if best_particle_loss < best_global_loss:
                best_global_loss = best_particle_loss
                best_particle = best_particles[best_loss_idx]

                best_extrinsics = RandomExtrinsicsGenerator.generate_view(
                    random_variable=best_particle.unsqueeze(0),
                    batch_size=1,
                    device=best_particle.device,
                ).squeeze()
                translation_error = torch.norm(
                    best_extrinsics[:3, 3]
                    - torch.tensor(gt_extrinsics[:3, 3], device=device)
                )
                best_loss_print = np.round(best_global_loss.item(), 3)
                translation_error_print = np.round(translation_error.item(), 2)

                rich.print(
                    f"New best loss: {best_loss_print}, particle idx: {best_loss_idx}, translation error: {translation_error_print}"
                )

    except KeyboardInterrupt:
        pass

    if visualize:
        plt.ioff()
        plt.show()

    best_extrinsics = RandomExtrinsicsGenerator.generate_view(
        random_variable=best_particle.unsqueeze(0),
        batch_size=1,
        device=best_particle.device,
    ).squeeze()

    # render best extrinsics (re-instantiate with batch size n_joint_configurations)
    meshes = rrd.TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name, visual=False
        ),
        batch_size=n_joint_configurations,
        device=device,
        target_reduction=target_reduction,
    )
    camera = rrd.VirtualCamera(
        resolution=(height, width),
        intrinsics=intrinsics,
        extrinsics=best_extrinsics,
        device=device,
    )
    scene = rrd.RobotScene(
        meshes=meshes,
        kinematics=kinematics,
        renderer=renderer,
        cameras={"camera": camera},
    )
    joint_states = joint_states[:n_joint_configurations]
    scene.configure_robot_joint_states(joint_states)
    renders = scene.observe_from("camera")
    for render in renders:
        cv2.imshow("best renders", render.cpu().numpy())
        cv2.waitKey(0)

    np.save("best_extrinsics.npy", best_extrinsics.cpu().numpy())


if __name__ == "__main__":
    test_particle_swarm()
"""
