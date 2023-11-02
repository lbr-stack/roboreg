import os
from typing import List

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import transformations as tf
import xacro
from ament_index_python import get_package_share_directory

from roboreg.o3d_robot import O3DRobot


def render(
    robot: O3DRobot,
    joint_state: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    width: int = 640,
    height: int = 360,
    material_color: List[float] = [1.0, 1.0, 1.0, 1.0],
    background_color: List[float] = [0.0, 0.0, 0.0, 1.0],
) -> np.ndarray:
    # re-configure robot
    robot.set_joint_positions(joint_state)

    # create rendering scene
    meshes = robot.meshes
    render = rendering.OffscreenRenderer(width, height)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = material_color
    mtl.shader = "defaultUnlit"
    render.scene.set_background(background_color)
    for idx, mesh in enumerate(meshes):
        render.scene.add_geometry(f"link_{idx}", mesh, mtl)

    # compute up eye center from extrinsic matrix
    up = -extrinsic_matrix[1, :3]
    eye = -np.linalg.inv(extrinsic_matrix[:3, :3]) @ extrinsic_matrix[:3, 3]
    center = eye + extrinsic_matrix[2, :3]

    print("center: ", center.transpose())
    print("eye: ", eye.transpose())
    print("up: ", up.transpose())

    render.setup_camera(
        intrinsic_matrix,
        np.eye(4),  # TODO: replace setup camera function or fix it
        width,
        height,
    )

    # render
    render.scene.camera.look_at(center, eye, up)
    o3d_render = render.render_to_image()
    return np.asarray(o3d_render)


if __name__ == "__main__":
    input_prefix = "/home/martin/Dev/records/23_10_05_base_to_base_reg/left_low_res"
    ht_file = "ht.npy"
    output_prefix = "/tmp/img"

    # load robot
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )

    robot = O3DRobot(urdf)

    # TODO: remove hardcoding and get proper intrinsics
    scale_height = 640.0 / 448.0
    scale_width = 360.0 / 256.0
    intrinsic_matrix = np.array(
        [
            [184.9792022705078 * scale_height, 0.0, 222.7788848876953 * scale_height],
            [0.0, 187.91539001464844 * scale_width, 123.90357971191406 * scale_width],
            [0.0, 0.0, 1.0],
        ]
    )

    for i in range(8):
        ###########
        # load data
        ###########
        joint_state_file = f"joint_state_{i}.npy"
        img_file = f"img_{i}.png"
        mask_file = f"mask_{i}.png"

        extrinsic_matrix = np.load(os.path.join(input_prefix, ht_file))
        joint_state = np.load(os.path.join(input_prefix, joint_state_file))
        img = cv2.imread(os.path.join(input_prefix, img_file))
        mask = cv2.imread(os.path.join(input_prefix, mask_file), cv2.IMREAD_GRAYSCALE)

        ########################
        # homogeneous -> optical
        ########################
        HT_cam_base = np.load(
            os.path.join(input_prefix, ht_file)
        )  # camera -> base frame (reference)
        HT_base_cam = np.linalg.inv(
            HT_cam_base
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
        o3d_render = render(
            robot=robot,
            joint_state=joint_state,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=HT_optical_base,
            width=640,
            height=360,
        )

        #################
        # post-processing
        #################
        o3d_render_gray = cv2.cvtColor(o3d_render, cv2.COLOR_RGB2GRAY)

        # pad zeros to mask
        o3d_render_blue = np.stack(
            [o3d_render_gray, np.zeros_like(o3d_render_gray), np.zeros_like(o3d_render_gray)],
            axis=2,
        )

        overlay = cv2.addWeighted(img, 1.0, o3d_render_blue, 1.0, 0)

        diff = np.abs(mask - o3d_render_gray)

        # resize to double size
        img = cv2.resize(img, [int(size * 4) for size in img.shape[:2][::-1]])
        mask = cv2.resize(mask, [int(size * 4) for size in mask.shape[:2][::-1]])
        o3d_render_gray = cv2.resize(
            o3d_render_gray, [int(size * 4) for size in o3d_render_gray.shape[:2][::-1]]
        )
        diff = cv2.resize(diff, [int(size * 4) for size in diff.shape[:2][::-1]])
        overlay = cv2.resize(
            overlay, [int(size * 4) for size in overlay.shape[:2][::-1]]
        )

        ######
        # save
        ######
        cv2.imwrite(os.path.join(output_prefix, f"img_{i}.jpg"), img)
        cv2.imwrite(os.path.join(output_prefix, f"mask_{i}.jpg"), mask)
        cv2.imwrite(os.path.join(output_prefix, f"mask_render_{i}.jpg"), o3d_render_gray)
        cv2.imwrite(os.path.join(output_prefix, f"diff_{i}.jpg"), diff)
        cv2.imwrite(os.path.join(output_prefix, f"overlay_{i}.jpg"), overlay)
