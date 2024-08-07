import argparse
import os
import pathlib

import cv2
import numpy as np
import transformations as tf


from roboreg.io import find_files, parse_camera_info
from roboreg.util import generate_o3d_robot, overlay_mask


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_path", type=str, required=True, help="Path to the images."
    )
    parser.add_argument(
        "--joint_states_path", type=str, required=True, help="Path to the joint states."
    )
    parser.add_argument(
        "--camera_info",
        type=str,
        required=True,
        help="Path to the camera parameters, <path_to>/camera_info.yaml.",
    )
    parser.add_argument(
        "--image_pattern",
        type=str,
        default="image_*.png",
        help="Image file pattern.",
    )
    parser.add_argument(
        "--joint_states_pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--ht",
        type=str,
        required=True,
        help="Homogeneous transform from base to camera frame, <path_to>/HT_hydra_robust.npy.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path.",
    )
    return parser.parse_args()


def main():
    args = args_factory()

    images_path = args.images_path
    joint_states_path = args.joint_states_path
    ht = args.ht
    output_path = pathlib.Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # load robot
    robot = generate_o3d_robot()

    # load camera info
    height, width, intrinsic_matrix = parse_camera_info(args.camera_info)

    # read files
    joint_states_files = find_files(joint_states_path, args.joint_states_pattern)
    img_files = find_files(images_path, args.image_pattern)

    ########################
    # homogeneous -> optical
    ########################
    HT_base_cam = np.load(ht)  # base frame (reference / world) -> camera

    # static transforms
    HT_cam_optical = tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # camera -> optical

    # base to optical frame
    HT_base_optical = HT_base_cam @ HT_cam_optical  # base frame -> optical
    HT_optical_base = np.linalg.inv(HT_base_optical)

    robot.setup_render(
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=HT_optical_base,
        width=width,
        height=height,
    )

    for joint_state_file, img_file in zip(joint_states_files, img_files):
        joint_state = np.load(os.path.join(joint_states_path, joint_state_file))
        img = cv2.imread(os.path.join(images_path, img_file))

        #############
        # render mask
        #############
        o3d_render = robot.render(joint_state)

        #################
        # post-processing
        #################
        o3d_render = cv2.cvtColor(o3d_render, cv2.COLOR_RGB2GRAY)

        rendered = overlay_mask(
            img,
            o3d_render,
            "b",
            0.8,
            1.0,
        )

        ######
        # save
        ######
        prefix = img_file.split(".")[0]
        suffix = img_file.split(".")[1]
        cv2.imwrite(
            os.path.join(
                str(output_path.absolute()), f"{prefix}_render_overlay.{suffix}"
            ),
            rendered,
        )


if __name__ == "__main__":
    main()
