import argparse
import importlib
import os
from typing import Tuple

import cv2
import numpy as np
import rich
import rich.progress
import torch

from roboreg import differentiable as rrd
from roboreg.io import find_files
from roboreg.util import overlay_mask


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
        "--epochs",
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
        cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        for file in left_mask_files
    ]

    images = np.array(images)
    joint_states = torch.tensor(
        np.array(joint_states), dtype=torch.float32, device=device
    )
    masks = (
        torch.tensor(np.array(masks), dtype=torch.float32, device=device).unsqueeze(-1)
        / 255.0
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
        device=device,
    )
    scene = rrd.robot_scene_factory(
        device=device,
        batch_size=joint_states.shape[0],
        ros_package=args.ros_package,
        xacro_path=args.xacro_path,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
        camera_info_files={
            "camera": args.camera_info_file,
        },
        extrinsics_files={
            "camera": args.extrinsics_file,
        },
        visual=args.visual_meshes,
    )

    # configure scene
    scene.configure_robot_joint_states(joint_states)

    # enable gradient tracking and instantiate optimizer
    scene.cameras["camera"].extrinsics.requires_grad = True
    optimizer = getattr(importlib.import_module("torch.optim"), args.optimizer)(
        [scene.cameras["camera"].extrinsics], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    metric = torch.nn.BCELoss()
    best_extrinsics = scene.cameras["camera"].extrinsics
    best_loss = float("inf")

    for _ in rich.progress.track(range(args.epochs), "Optimizing..."):
        if not scene.cameras["camera"].extrinsics.requires_grad:
            raise ValueError("Extrinsics require gradients.")
        if not torch.is_grad_enabled():
            raise ValueError("Gradients must be enabled.")
        renders = {
            "camera": scene.observe_from("camera"),
        }
        loss = metric(renders["camera"], masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        rich.print(
            f"Loss: {np.round(loss.item(), 3)}, best loss: {np.round(best_loss, 3)}, lr: {scheduler.get_last_lr().pop()}"
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_extrinsics = scene.cameras["camera"].extrinsics.detach().clone()

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

    with torch.no_grad():
        # render final results and save extrinsics
        scene.cameras["camera"].extrinsics = best_extrinsics
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
        scene.cameras["camera"].extrinsics.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
