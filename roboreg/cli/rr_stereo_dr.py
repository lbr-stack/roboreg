import argparse
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
        "--device",
        type=str,
        default="cuda",
        help="Device to use, e.g. 'cuda' or 'cpu'.",
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
        "--display_progress",
        action="store_true",
        help="Display optimization progress.",
    )
    parser.add_argument(
        "--ros_package",
        type=str,
        default="lbr_description",
        help="Package where the URDF is located.",
    )
    parser.add_argument(
        "--xacro_path",
        type=str,
        default="urdf/med7/med7.xacro",
        help="Path to the xacro file, relative to --ros_package.",
    )
    parser.add_argument(
        "--root_link_name", type=str, default="link_0", help="Root link name."
    )
    parser.add_argument(
        "--end_link_name", type=str, default="link_7", help="End link name."
    )
    parser.add_argument(
        "--left_camera_info_file",
        type=str,
        required=True,
        help="Path to the left camera parameters, <path_to>/left_camera_info.yaml.",
    )
    parser.add_argument(
        "--right_camera_info_file",
        type=str,
        required=True,
        help="Path to the left camera parameters, <path_to>/right_camera_info.yaml.",
    )
    parser.add_argument(
        "--left_extrinsics_file",
        type=str,
        required=True,
        help="Homogeneous transforms from base to left camera frame, <path_to>/HT_hydra_robust.npy.",
    )
    parser.add_argument(
        "--right_extrinsics_file",
        type=str,
        required=True,
        help="Homogeneous transforms from base to right camera frame, <path_to>/HT_right_to_left.npy.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--left_image_pattern",
        type=str,
        default="left_image_*.png",
        help="Left image file pattern.",
    )
    parser.add_argument(
        "--right_image_pattern",
        type=str,
        default="right_image_*.png",
        help="Right image file pattern.",
    )
    parser.add_argument(
        "--joint_states_pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--left_mask_pattern",
        type=str,
        default="left_mask_*.png",
        help="Left mask file pattern.",
    )
    parser.add_argument(
        "--right_mask_pattern",
        type=str,
        default="right_mask_*.png",
        help="Right mask file pattern.",
    )
    parser.add_argument(
        "--left_output_file",
        type=str,
        default="HT_left_dr.npy",
        help="Left output file name. Relative to the path.",
    )
    parser.add_argument(
        "--right_output_file",
        type=str,
        default="HT_right_dr.npy",
        help="Right output file name. Relative to the path.",
    )
    return parser.parse_args()


def parse_data(
    path: str,
    left_image_pattern: str,
    right_image_pattern: str,
    joint_states_pattern: str,
    left_mask_pattern: str,
    right_mask_pattern: str,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, torch.FloatTensor, torch.FloatTensor]:
    left_image_files = find_files(path, left_image_pattern)
    right_image_files = find_files(path, right_image_pattern)
    joint_states_files = find_files(path, joint_states_pattern)
    left_mask_files = find_files(path, left_mask_pattern)
    right_mask_files = find_files(path, right_mask_pattern)

    rich.print("Found the following files:")
    rich.print(f"Left images: {left_image_files}")
    rich.print(f"Right images: {right_image_files}")
    rich.print(f"Joint states: {joint_states_files}")
    rich.print(f"Left masks: {left_mask_files}")
    rich.print(f"Right masks: {right_mask_files}")

    if (
        len(left_image_files) != len(right_image_files)
        or len(left_image_files) != len(joint_states_files)
        or len(left_image_files) != len(left_mask_files)
        or len(left_image_files) != len(right_mask_files)
    ):
        raise ValueError(
            "Number of left / right images, joint states, left / right masks do not match."
        )

    left_images = [cv2.imread(os.path.join(path, file)) for file in left_image_files]
    right_images = [cv2.imread(os.path.join(path, file)) for file in right_image_files]
    joint_states = [np.load(os.path.join(path, file)) for file in joint_states_files]
    left_masks = [
        cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        for file in left_mask_files
    ]
    right_masks = [
        cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        for file in right_mask_files
    ]

    left_images = np.array(left_images)
    right_images = np.array(right_images)
    joint_states = torch.tensor(
        np.array(joint_states), dtype=torch.float32, device=device
    )
    left_masks = (
        torch.tensor(
            np.array(left_masks), dtype=torch.float32, device=device
        ).unsqueeze(-1)
        / 255.0
    )
    right_masks = (
        torch.tensor(
            np.array(right_masks), dtype=torch.float32, device=device
        ).unsqueeze(-1)
        / 255.0
    )
    return left_images, right_images, joint_states, left_masks, right_masks


def main() -> None:
    args = args_factory()
    left_images, right_images, joint_states, left_masks, right_masks = parse_data(
        path=args.path,
        left_image_pattern=args.left_image_pattern,
        right_image_pattern=args.right_image_pattern,
        joint_states_pattern=args.joint_states_pattern,
        left_mask_pattern=args.left_mask_pattern,
        right_mask_pattern=args.right_mask_pattern,
        device=args.device,
    )
    scene = rrd.robot_scene_factory(
        device=args.device,
        batch_size=joint_states.shape[0],
        ros_package=args.ros_package,
        xacro_path=args.xacro_path,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
        camera_info_files={
            "left": args.left_camera_info_file,
            "right": args.right_camera_info_file,
        },
        extrinsics_files={
            "left": args.left_extrinsics_file,
            "right": args.right_extrinsics_file,
        },
    )

    # configure scene
    scene.configure_robot_joint_states(joint_states)

    # enable gradient tracking and instantiate optimizer
    scene.cameras["left"].extrinsics.requires_grad = True
    optimizer = torch.optim.SGD([scene.cameras["left"].extrinsics], lr=args.lr)
    metric = torch.nn.BCELoss()
    best_extrinsics = scene.cameras["left"].extrinsics.detach().clone()
    best_loss = float("inf")

    for _ in rich.progress.track(range(args.epochs), "Optimizing..."):
        renders = {
            "left": scene.observe_from("left"),
            "right": scene.observe_from("right", scene.cameras["left"].extrinsics),
        }
        loss = metric(renders["left"], left_masks) + metric(
            renders["right"], right_masks
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rich.print(f"Loss: {loss.item()}")

        if loss < best_loss:
            best_loss = loss
            best_extrinsics = scene.cameras["left"].extrinsics.detach().clone()

        # display optimization progress
        if args.display_progress:
            overlays = []
            left_render = renders["left"][0].squeeze().detach().cpu().numpy()
            left_image = left_images[0]
            overlays.append(
                overlay_mask(
                    left_image,
                    (left_render * 255.0).astype(np.uint8),
                    scale=1.0,
                )
            )
            right_render = renders["right"][0].squeeze().detach().cpu().numpy()
            right_image = right_images[0]
            overlays.append(
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
                        np.abs(left_render - left_masks[0].squeeze().cpu().numpy()),
                        cv2.COLOR_GRAY2BGR,
                    )
                    * 255.0
                ).astype(np.uint8)
            )
            differences.append(
                (
                    cv2.cvtColor(
                        np.abs(right_render - right_masks[0].squeeze().cpu().numpy()),
                        cv2.COLOR_GRAY2BGR,
                    )
                    * 255.0
                ).astype(np.uint8)
            )
            cv2.imshow(
                "top: overlays, bottom: differences, left: left view, right: right view",
                cv2.resize(
                    np.vstack([np.hstack(overlays), np.hstack(differences)]),
                    (0, 0),
                    fx=0.5,
                    fy=0.5,
                ),
            )
            cv2.waitKey(1)

    # render final results and save extrinsics
    scene.cameras["left"].extrinsics = best_extrinsics
    renders = {
        "left": scene.observe_from("left"),
        "right": scene.observe_from("right", scene.cameras["left"].extrinsics),
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
        left_difference = np.abs(left_render - left_masks[i].squeeze().cpu().numpy())
        right_difference = np.abs(right_render - right_masks[i].squeeze().cpu().numpy())

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
        scene.cameras["left"].extrinsics.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(args.path, args.right_output_file),
        scene.cameras["left"].extrinsics.detach().cpu().numpy()
        @ scene.cameras["right"].extrinsics.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
