import argparse
import os

import cv2
import numpy as np
import transformations as tf

from roboreg.util import find_files, generate_o3d_robot, overlay_mask, parse_camera_info


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the images.")
    parser.add_argument(
        "--camera_info",
        type=str,
        default="camera_info.yaml",
        help="Path to the camera parameters.",
    )
    parser.add_argument(
        "--image_pattern",
        type=str,
        default="image_*.png",
        help="Image file pattern.",
    )
    parser.add_argument(
        "--mask_pattern", type=str, default="mask_*.png", help="Mask file pattern."
    )
    parser.add_argument(
        "--joint_states_pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--ht_file",
        type=str,
        default="HT_hydra_robust.npy",
        help="Homogeneous transform from base to camera frame.",
    )
    return parser.parse_args()


def main():
    args = args_factory()

    path = args.path
    ht_file = args.ht_file
    output_path = path

    # load robot
    robot = generate_o3d_robot()

    # load camera info
    height, width, intrinsic_matrix = parse_camera_info(
        os.path.join(path, args.camera_info)
    )

    # read files
    joint_states_files = find_files(path, args.joint_states_pattern)
    img_files = find_files(path, args.image_pattern)
    mask_files = find_files(path, args.mask_pattern)

    for joint_state_file, img_file, mask_file in zip(
        joint_states_files, img_files, mask_files
    ):
        joint_state = np.load(os.path.join(path, joint_state_file))
        img = cv2.imread(os.path.join(path, img_file))
        mask = cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE)

        ########################
        # homogeneous -> optical
        ########################
        HT_base_cam = np.load(
            os.path.join(path, ht_file)
        )  # base frame (reference / world) -> camera

        # static transforms
        HT_cam_optical = tf.quaternion_matrix(
            [0.5, -0.5, 0.5, -0.5]
        )  # camera -> optical

        # base to optical frame
        HT_base_optical = HT_base_cam @ HT_cam_optical  # base frame -> optical
        HT_optical_base = np.linalg.inv(HT_base_optical)

        #############
        # render mask
        #############
        robot.set_joint_positions(joint_state)
        o3d_render = robot.render(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=HT_optical_base,
            width=width,
            height=height,
        )

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

        masked = overlay_mask(
            img,
            mask,
            "g",
            0.8,
            1.0,
        )

        ######
        # save
        ######
        cv2.imwrite(
            os.path.join(output_path, img_file.replace("img", "rendered")),
            rendered,
        )
        cv2.imwrite(
            os.path.join(output_path, img_file.replace("img", "masked")),
            masked,
        )


if __name__ == "__main__":
    main()
