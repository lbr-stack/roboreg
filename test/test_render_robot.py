import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import transformations as tf

from roboreg.util import generate_o3d_robot, parse_camera_info, find_files


def test_render_robot():
    input_prefix = "test/data/lbr_med7/high_res"
    ht_file = "HT_base_cam_optimal_new.npy"
    output_prefix = input_prefix

    # load robot
    robot = generate_o3d_robot()

    # camera intrinsics
    height, width, intrinsic_matrix = parse_camera_info(
        os.path.join(input_prefix, "left_camera_info.yaml")
    )

    joint_state_files = find_files(input_prefix, "joint_state_*.npy")
    img_files = find_files(input_prefix, "img_*.png")
    mask_files = find_files(input_prefix, "mask_*.png")

    for joint_state_file, img_file, mask_file in zip(
        joint_state_files, img_files, mask_files
    ):
        joint_state = np.load(os.path.join(input_prefix, joint_state_file))
        img = cv2.imread(os.path.join(input_prefix, img_file))
        mask = cv2.imread(os.path.join(input_prefix, mask_file), cv2.IMREAD_GRAYSCALE)

        ########################
        # homogeneous -> optical
        ########################
        HT_base_cam = np.load(
            os.path.join(input_prefix, ht_file)
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
        o3d_render_gray = cv2.cvtColor(o3d_render, cv2.COLOR_RGB2GRAY)

        # pad zeros to mask
        o3d_render_blue = np.stack(
            [
                o3d_render_gray,
                np.zeros_like(o3d_render_gray),
                np.zeros_like(o3d_render_gray),
            ],
            axis=2,
        )

        mask_render_green = np.stack(
            [np.zeros_like(mask), mask, np.zeros_like(mask)], axis=2
        )

        overlay_img_render = cv2.addWeighted(img, 1.0, o3d_render_blue, 1.0, 0)
        overlay_img_mask = cv2.addWeighted(img, 1.0, mask_render_green, 1.0, 0)

        # resize to double size
        scale = 2.0
        overlay_img_render = cv2.resize(
            overlay_img_render,
            [int(size * scale) for size in overlay_img_render.shape[:2][::-1]],
        )
        overlay_img_mask = cv2.resize(
            overlay_img_mask,
            [int(size * scale) for size in overlay_img_mask.shape[:2][::-1]],
        )

        ######
        # save
        ######
        cv2.imwrite(
            os.path.join(output_prefix, img_file.replace("img", "overlay_render")),
            overlay_img_render,
        )
        cv2.imwrite(
            os.path.join(output_prefix, img_file.replace("img", "overlay_mask")),
            overlay_img_mask,
        )


if __name__ == "__main__":
    test_render_robot()
