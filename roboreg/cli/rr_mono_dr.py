import argparse
import importlib
import os
from typing import Tuple

import cv2
import numpy as np
import pytorch_kinematics as pk
import rich
import rich.progress
import torch

from roboreg.io import find_files
from roboreg.losses import soft_dice_loss
from roboreg.util import mask_exponential_distance_transform, overlay_mask
from roboreg.util.factories import create_robot_scene, create_virtual_camera


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--sigma",
        type=float,
        default=2.0,
        help="Sigma for the exponential distance transform on target masks.",
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
        "--visual-meshes",
        action="store_true",
        help="If set, visual meshes will be used instead of collision meshes.",
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
    return parser.parse_args()


def parse_data(
    path: str,
    image_pattern: str,
    joint_states_pattern: str,
    mask_pattern: str,
    sigma: float = 2.0,
    device: str = "cuda",
) -> Tuple[np.ndarray, torch.FloatTensor, torch.FloatTensor]:
    image_files = find_files(path, image_pattern)
    joint_states_files = find_files(path, joint_states_pattern)
    left_mask_files = find_files(path, mask_pattern)

    rich.print("Found the following files:")
    rich.print(f"Images: {image_files}")
    rich.print(f"Joint states: {joint_states_files}")
    rich.print(f"Masks: {left_mask_files}")

    if len(image_files) != len(joint_states_files) or len(image_files) != len(
        left_mask_files
    ):
        raise ValueError("Number of images, joint states, masks do not match.")

    images = [cv2.imread(os.path.join(path, file)) for file in image_files]
    joint_states = [np.load(os.path.join(path, file)) for file in joint_states_files]
    masks = [
        mask_exponential_distance_transform(
            cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE), sigma=sigma
        )
        for file in left_mask_files
    ]

    images = np.array(images)
    joint_states = torch.tensor(
        np.array(joint_states), dtype=torch.float32, device=device
    )
    masks = torch.tensor(np.array(masks), dtype=torch.float32, device=device).unsqueeze(
        -1
    )
    return images, joint_states, masks


def main() -> None:
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, joint_states, masks = parse_data(
        path=args.path,
        image_pattern=args.image_pattern,
        joint_states_pattern=args.joint_states_pattern,
        mask_pattern=args.mask_pattern,
        sigma=args.sigma,
        device=device,
    )

    # instantiate camera with default identity extrinsics because we optimize for robot pose instead
    camera = {
        "camera": create_virtual_camera(
            camera_info_file=args.camera_info_file,
            device=device,
        )
    }

    # instantiate robot scene
    scene = create_robot_scene(
        batch_size=joint_states.shape[0],
        ros_package=args.ros_package,
        xacro_path=args.xacro_path,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
        visual=args.visual_meshes,
        cameras=camera,
        device=device,
    )

    # load extrinsics estimate
    extrinsics = torch.tensor(
        np.load(args.extrinsics_file), dtype=torch.float32, device=device
    )
    extrinsics_inv = torch.linalg.inv(extrinsics)

    # enable gradient tracking and instantiate optimizer
    extrinsics_9d_inv = pk.matrix44_to_se3_9d(extrinsics_inv)
    extrinsics_9d_inv.requires_grad = True
    optimizer = getattr(importlib.import_module("torch.optim"), args.optimizer)(
        [extrinsics_9d_inv], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    best_extrinsics = extrinsics
    best_extrinsics_inv = extrinsics_inv
    best_loss = float("inf")

    for iteration in rich.progress.track(
        range(1, args.max_iterations + 1), "Optimizing..."
    ):
        if not extrinsics_9d_inv.requires_grad:
            raise ValueError("Extrinsics require gradients.")
        if not torch.is_grad_enabled():
            raise ValueError("Gradients must be enabled.")
        extrinsics_inv = pk.se3_9d_to_matrix44(extrinsics_9d_inv)
        scene.robot.configure(joint_states, extrinsics_inv)
        renders = {
            "camera": scene.observe_from("camera"),
        }
        loss = soft_dice_loss(renders["camera"], masks).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        rich.print(
            f"Step [{iteration} / {args.max_iterations}], loss: {np.round(loss.item(), 3)}, best loss: {np.round(best_loss, 3)}, lr: {scheduler.get_last_lr().pop()}"
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_extrinsics_inv = extrinsics_inv.detach().clone()
            best_extrinsics = torch.linalg.inv(best_extrinsics_inv)

        # display optimization progress
        if args.display_progress:
            render = renders["camera"][0].squeeze().detach().cpu().numpy()
            image = images[0]
            render_overlay = overlay_mask(
                image,
                (render * 255.0).astype(np.uint8),
                scale=1.0,
            )
            # difference left / right render / mask
            difference = (
                cv2.cvtColor(
                    np.abs(render - masks[0].squeeze().cpu().numpy()),
                    cv2.COLOR_GRAY2BGR,
                )
                * 255.0
            ).astype(np.uint8)
            # overlay segmentation mask
            segmentation_overlay = overlay_mask(
                image,
                (masks[0].squeeze().cpu().numpy() * 255.0).astype(np.uint8),
                mode="b",
                scale=1.0,
            )
            cv2.imshow(
                "left to right: render overlay, difference, segmentation overlay",
                cv2.resize(
                    np.hstack(
                        [
                            render_overlay,
                            difference,
                            segmentation_overlay,
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
        scene.robot.configure(joint_states, best_extrinsics_inv)
        renders = scene.observe_from("camera")

    for i, render in enumerate(renders):
        render = render.squeeze().cpu().numpy()
        overlay = overlay_mask(images[i], (render * 255.0).astype(np.uint8), scale=1.0)
        difference = np.abs(render - masks[i].squeeze().cpu().numpy())

        cv2.imwrite(os.path.join(args.path, f"dr_overlay_{i}.png"), overlay)
        cv2.imwrite(
            os.path.join(args.path, f"dr_difference_{i}.png"),
            (difference * 255.0).astype(np.uint8),
        )

    np.save(
        os.path.join(args.path, args.output_file),
        best_extrinsics.cpu().numpy(),
    )


if __name__ == "__main__":
    main()
