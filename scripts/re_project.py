import os
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import transformations as tf
import xacro
from ament_index_python import get_package_share_directory

from roboreg.o3d_robot import O3DRobot

# pose
# mesh
# camera matrix


def render(
    robot: O3DRobot,
    width: int = 640,
    height: int = 360,
    path: str = "/home/martin/Dev/records/23_10_05_base_to_base_reg/left_low_res",
    ht_file: str = "ht.npy",
    joint_state_file: str = "joint_state_0.npy",
    mask_file: str = "mask_0.png",
    img_file: str = "img_0.png",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # re-configure robot
    joint_state = np.load(os.path.join(path, joint_state_file))
    robot.set_joint_positions(joint_state)

    # robot.visualize_point_clouds()

    # render meshes
    meshes = robot.meshes

    render = rendering.OffscreenRenderer(width, height)

    # pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     width=width,
    #     height=height,
    #     fx=640,
    #     fy=640,
    #     cx=320,
    #     cy=320,
    # )

    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
    for idx, mesh in enumerate(meshes):
        render.scene.add_geometry(f"link_{idx}", mesh, mtl)

    # homogeneous -> optical -> look at
    HT_cam_base = np.load(
        os.path.join(path, ht_file)
    )  # camera -> base frame (reference)
    HT_base_cam = np.linalg.inv(HT_cam_base)  # base frame (reference / world) -> camera
    print("HT_base_cam:\n", HT_base_cam)
    print("quat HT_base_cam:\n", tf.quaternion_from_matrix(HT_base_cam))
    print("trans HT_base_cam:\n", HT_base_cam[:3, 3])

    # static transforms
    HT_cam_optical = tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # camera -> optical

    # base to optical frame
    HT_base_optical = HT_base_cam @ HT_cam_optical  # base frame -> optical

    print("HT_base_optical:\n", HT_base_optical)
    print("quat HT_base_optical:\n", tf.quaternion_from_matrix(HT_base_optical))
    print("trans HT_base_optical:\n", HT_base_optical[:3, 3])

    # HT to look at... meshrender https://github.com/BerkeleyAutomation/meshrender
    HT_optical_base = np.linalg.inv(HT_base_optical)
    up = -HT_optical_base[1, :3]
    eye = -np.linalg.inv(HT_optical_base[:3, :3]) @ HT_optical_base[:3, 3]

    center = eye + HT_optical_base[2, :3]

    print("center: ", center.transpose())
    print("eye: ", eye.transpose())
    print("up: ", up.transpose())

    scale_height = 640.0 / 448.0
    scale_width = 360.0 / 256.0
    intrinsic_matrix = np.array(
        [
            [184.9792022705078 * scale_height, 0.0, 222.7788848876953 * scale_height],
            [0.0, 187.91539001464844 * scale_width, 123.90357971191406 * scale_width],
            [0.0, 0.0, 1.0],
        ]
    )

    render.setup_camera(
        intrinsic_matrix,
        HT_base_optical,
        width,
        height,
    )

    # look at
    render.scene.camera.look_at(center, eye, up)

    img_o3d = render.render_to_image()

    # we can now save the rendered image right at this point
    o3d.io.write_image("output.png", img_o3d, 9)

    # plot diff
    mask = cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE)
    mask_render = cv2.cvtColor(np.asarray(img_o3d), cv2.COLOR_RGB2GRAY)
    mask_render = np.where(mask_render > 128, 255, 0).astype(np.uint8)

    img = cv2.imread(os.path.join(path, img_file))

    # pad zeros to mask
    mask_color_render = np.stack(
        [mask_render, np.zeros_like(mask_render), np.zeros_like(mask_render)], axis=2
    )

    overlay = cv2.addWeighted(img, 1.0, mask_color_render, 1.0, 0)

    return img, mask, mask_render, overlay


if __name__ == "__main__":
    output_prefix = "/tmp/img"

    # load robot
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )

    robot = O3DRobot(urdf)

    for i in range(8):
        img, mask, mask_render, overlay = render(
            robot=robot,
            width=640,
            height=360,
            path="/home/martin/Dev/records/23_10_05_base_to_base_reg/left_low_res",
            ht_file="ht.npy",
            joint_state_file=f"joint_state_{i}.npy",
            mask_file=f"mask_{i}.png",
            img_file=f"img_{i}.png",
        )

        diff = np.abs(mask - mask_render)

        # resize to double size
        img = cv2.resize(img, [int(size * 4) for size in img.shape[:2][::-1]])
        mask = cv2.resize(mask, [int(size * 4) for size in mask.shape[:2][::-1]])
        mask_render = cv2.resize(
            mask_render, [int(size * 4) for size in mask_render.shape[:2][::-1]]
        )
        diff = cv2.resize(diff, [int(size * 4) for size in diff.shape[:2][::-1]])
        overlay = cv2.resize(
            overlay, [int(size * 4) for size in overlay.shape[:2][::-1]]
        )

        cv2.imwrite(os.path.join(output_prefix, f"img_{i}.jpg"), img)
        cv2.imwrite(os.path.join(output_prefix, f"mask_{i}.jpg"), mask)
        cv2.imwrite(os.path.join(output_prefix, f"mask_render_{i}.jpg"), mask_render)
        cv2.imwrite(os.path.join(output_prefix, f"diff_{i}.jpg"), diff)
        cv2.imwrite(os.path.join(output_prefix, f"overlay_{i}.jpg"), overlay)
        cv2.waitKey()
