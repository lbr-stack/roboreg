import os

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import transformations as tf
import xacro
from ament_index_python import get_package_share_directory
import cv2

from roboreg.o3d_robot import O3DRobot

# pose
# mesh
# camera matrix


def main() -> None:
    width = 640  # same as mask
    height = 360
    path = "/home/martin/Dev/records/23_10_05_base_to_base_reg/left_low_res"
    ht_file = "ht.npy"
    joint_state_file = "joint_state_0.npy"
    mask_file = "mask_0.png"
    img_file = "img_0.png"

    # load robot
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )

    robot = O3DRobot(urdf)

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
    mask_render = np.where(mask_render > 10, 255, 0).astype(np.uint8)

    img = cv2.imread(os.path.join(path, img_file))

    # pad zeros to mask
    mask_render = np.stack(
        [mask_render, np.zeros_like(mask_render), np.zeros_like(mask_render)], axis=2
    )

    overlay = cv2.addWeighted(img, 1.0, mask_render, 1.0, 0)

    cv2.imshow("overlay", overlay)
    cv2.waitKey()


if __name__ == "__main__":
    main()
