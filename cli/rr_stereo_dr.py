import argparse
import importlib
import os
from enum import Enum

import cv2
import numpy as np
import pytorch_kinematics as pk
import rich
import rich.progress
import torch

from roboreg.io import find_files, parse_stereo_data
from roboreg.losses import soft_dice_loss
from roboreg.util import mask_distance_transform, mask_exponential_decay, overlay_mask
from roboreg.util.factories import create_robot_scene, create_virtual_camera


class REGISTRATION_MODE(Enum):
    DISTANCE_FUNCTION = "distance-function"
    SEGMENTATION = "segmentation"


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer to use, e.g. 'Adam' or 'SGD'. Imported from torch.optim.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Number of epochs to optimize for.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=100,
        help="Step size for the learning rate scheduler.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma for the learning rate scheduler.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in REGISTRATION_MODE],
        default=REGISTRATION_MODE.DISTANCE_FUNCTION.value,
        help="Registration mode.",
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
        "--collision-meshes",
        action="store_true",
        help="If set, collision meshes will be used instead of visual meshes.",
    )
    parser.add_argument(
        "--left-camera-info-file",
        type=str,
        required=True,
        help="Full path to left camera parameters, <path_to>/left_camera_info.yaml.",
    )
    parser.add_argument(
        "--right-camera-info-file",
        type=str,
        required=True,
        help="Full path to right camera parameters, <path_to>/right_camera_info.yaml.",
    )
    parser.add_argument(
        "--left-extrinsics-file",
        type=str,
        required=True,
        help="Full path to homogeneous transforms from base to left camera frame, <path_to>/HT_hydra_robust.npy.",
    )
    parser.add_argument(
        "--right-extrinsics-file",
        type=str,
        required=True,
        help="Full path to homogeneous transforms from base to right camera frame, <path_to>/HT_right_to_left.npy.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--left-image-pattern",
        type=str,
        default="left_image_*.png",
        help="Left image file pattern.",
    )
    parser.add_argument(
        "--right-image-pattern",
        type=str,
        default="right_image_*.png",
        help="Right image file pattern.",
    )
    parser.add_argument(
        "--joint-states-pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--left-mask-pattern",
        type=str,
        default="left_mask_*.png",
        help="Left mask file pattern.",
    )
    parser.add_argument(
        "--right-mask-pattern",
        type=str,
        default="right_mask_*.png",
        help="Right mask file pattern.",
    )
    parser.add_argument(
        "--left-output-file",
        type=str,
        default="HT_left_dr.npy",
        help="Left output file name. Relative to --path.",
    )
    parser.add_argument(
        "--right-output-file",
        type=str,
        default="HT_right_dr.npy",
        help="Right output file name. Relative to --path.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=2,
        help="Number of concurrent compilation jobs for nvdiffrast. Only relevant on first run.",
    )
    return parser.parse_args()


def main() -> None:
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(args.max_jobs)  # limit number of concurrent jobs
    mode = REGISTRATION_MODE(args.mode)

    # load data
    left_image_files = find_files(args.path, args.left_image_pattern)
    right_image_files = find_files(args.path, args.right_image_pattern)
    joint_states_files = find_files(args.path, args.joint_states_pattern)
    left_mask_files = find_files(args.path, args.left_mask_pattern)
    right_mask_files = find_files(args.path, args.right_mask_pattern)
    left_images, right_images, joint_states, left_masks, right_masks = (
        parse_stereo_data(
            path=args.path,
            left_image_files=left_image_files,
            right_image_files=right_image_files,
            joint_states_files=joint_states_files,
            left_mask_files=left_mask_files,
            right_mask_files=right_mask_files,
        )
    )

    # pre-process data
    joint_states = torch.tensor(
        np.array(joint_states), dtype=torch.float32, device=device
    )
    if mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
        left_targets = [mask_distance_transform(mask) for mask in left_masks]
        right_targets = [mask_distance_transform(mask) for mask in right_masks]
    elif mode == REGISTRATION_MODE.SEGMENTATION:
        left_targets = [mask_exponential_decay(mask) for mask in left_masks]
        right_targets = [mask_exponential_decay(mask) for mask in right_masks]
    else:
        raise ValueError("Invalid registration mode.")
    left_targets = torch.tensor(
        np.array(left_targets), dtype=torch.float32, device=device
    ).unsqueeze(-1)
    right_targets = torch.tensor(
        np.array(right_targets), dtype=torch.float32, device=device
    ).unsqueeze(-1)

    # instantiate:
    #   - left camera with default identity extrinsics because we optimize for robot pose instead
    #   - right camera with transformation to left camera frame
    cameras = {
        "left": create_virtual_camera(
            camera_info_file=args.left_camera_info_file,
            device=device,
        ),
        "right": create_virtual_camera(
            camera_info_file=args.right_camera_info_file,
            extrinsics_file=args.right_extrinsics_file,
            device=device,
        ),
    }

    # instantiate robot scene
    scene = create_robot_scene(
        batch_size=joint_states.shape[0],
        ros_package=args.ros_package,
        xacro_path=args.xacro_path,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
        cameras=cameras,
        device=device,
        collision=args.collision_meshes,
    )

    # load extrinscis estimate......
    left_extrinsics = torch.tensor(
        np.load(args.left_extrinsics_file), dtype=torch.float32, device=device
    )
    left_extrinsics_inv = torch.linalg.inv(left_extrinsics)

    # enable gradient tracking and instantiate optimizer
    left_extrinsics_9d_inv = pk.matrix44_to_se3_9d(left_extrinsics_inv)
    left_extrinsics_9d_inv.requires_grad = True
    optimizer = getattr(importlib.import_module("torch.optim"), args.optimizer)(
        [left_extrinsics_9d_inv], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    best_left_extrinsics = left_extrinsics
    best_left_extrinsics_inv = left_extrinsics_inv
    best_loss = float("inf")

    for iteration in rich.progress.track(
        range(1, args.max_iterations + 1), "Optimizing..."
    ):
        if not left_extrinsics_9d_inv.requires_grad:
            raise ValueError("Extrinsics require gradients.")
        if not torch.is_grad_enabled():
            raise ValueError("Gradients must be enabled.")
        left_extrinsics_inv = pk.se3_9d_to_matrix44(left_extrinsics_9d_inv)
        scene.robot.configure(joint_states, left_extrinsics_inv)
        renders = {
            "left": scene.observe_from("left"),
            "right": scene.observe_from("right"),
        }
        if mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
            loss = torch.nn.functional.mse_loss(
                left_targets, renders["left"]
            ) + torch.nn.functional.mse_loss(right_targets, renders["right"])
        elif mode == REGISTRATION_MODE.SEGMENTATION:
            loss = (
                soft_dice_loss(left_targets, renders["left"]).mean()
                + soft_dice_loss(right_targets, renders["right"]).mean()
            )
        else:
            raise ValueError("Invalid registration mode.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        rich.print(
            f"Step [{iteration} / {args.max_iterations}], loss: {np.round(loss.item(), 3)}, best loss: {np.round(best_loss, 3)}, lr: {scheduler.get_last_lr().pop()}"
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_left_extrinsics_inv = left_extrinsics_inv.detach().clone()
            best_left_extrinsics = torch.linalg.inv(best_left_extrinsics_inv)

        # display optimization progress
        if args.display_progress:
            render_overlays = []
            left_render = renders["left"][0].squeeze().detach().cpu().numpy()
            left_image = left_images[0]
            render_overlays.append(
                overlay_mask(
                    left_image,
                    (left_render * 255.0).astype(np.uint8),
                    scale=1.0,
                )
            )
            right_render = renders["right"][0].squeeze().detach().cpu().numpy()
            right_image = right_images[0]
            render_overlays.append(
                overlay_mask(
                    right_image,
                    (right_render * 255.0).astype(np.uint8),
                    scale=1.0,
                )
            )
            # difference left / right render / mask
            differences = []
            differences.append(
                (
                    cv2.cvtColor(
                        np.abs(left_render - left_masks[0].astype(np.float32) / 255.0),
                        cv2.COLOR_GRAY2BGR,
                    )
                    * 255.0
                ).astype(np.uint8)
            )
            differences.append(
                (
                    cv2.cvtColor(
                        np.abs(
                            right_render - right_masks[0].astype(np.float32) / 255.0
                        ),
                        cv2.COLOR_GRAY2BGR,
                    )
                    * 255.0
                ).astype(np.uint8)
            )
            # overlay segmentation mask
            segmentation_overlays = []
            segmentation_overlays.append(
                overlay_mask(
                    left_image,
                    left_masks[0],
                    mode="b",
                    scale=1.0,
                )
            )
            segmentation_overlays.append(
                overlay_mask(
                    right_image,
                    right_masks[0],
                    mode="b",
                    scale=1.0,
                )
            )
            cv2.imshow(
                "top to bottom: render overlays, differences, segmentation overlays | left: left view, right: right view",
                cv2.resize(
                    np.vstack(
                        [
                            np.hstack(render_overlays),
                            np.hstack(differences),
                            np.hstack(segmentation_overlays),
                        ]
                    ),
                    (0, 0),
                    fx=0.5,
                    fy=0.5,
                ),
            )
            cv2.waitKey(1)

    # render final results and save extrinsics
    with torch.no_grad():
        scene.robot.configure(joint_states, best_left_extrinsics_inv)
        renders = {
            "left": scene.observe_from("left"),
            "right": scene.observe_from("right"),
        }

    for i, (left_render, right_render) in enumerate(
        zip(renders["left"], renders["right"])
    ):
        left_render = left_render.squeeze().cpu().numpy()
        right_render = right_render.squeeze().cpu().numpy()
        left_overlay = overlay_mask(
            left_images[i], (left_render * 255.0).astype(np.uint8), scale=1.0
        )
        right_overlay = overlay_mask(
            right_images[i], (right_render * 255.0).astype(np.uint8), scale=1.0
        )
        left_difference = np.abs(left_render - left_masks[i].astype(np.float32) / 255.0)
        right_difference = np.abs(
            right_render - right_masks[i].astype(np.float32) / 255.0
        )

        cv2.imwrite(os.path.join(args.path, f"left_dr_overlay_{i}.png"), left_overlay)
        cv2.imwrite(os.path.join(args.path, f"right_dr_overlay_{i}.png"), right_overlay)
        cv2.imwrite(
            os.path.join(args.path, f"left_dr_difference_{i}.png"),
            (left_difference * 255.0).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(args.path, f"right_dr_difference_{i}.png"),
            (right_difference * 255.0).astype(np.uint8),
        )

    np.save(
        os.path.join(args.path, args.left_output_file),
        best_left_extrinsics.cpu().numpy(),
    )
    np.save(
        os.path.join(args.path, args.right_output_file),
        best_left_extrinsics.cpu().numpy()
        @ scene.cameras["right"].extrinsics.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
