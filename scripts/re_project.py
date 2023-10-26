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

    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
    for idx, mesh in enumerate(meshes):
        render.scene.add_geometry(f"link_{idx}", mesh, mtl)

    # homogeneous -> optical -> look at
    HT = np.load(os.path.join(path, ht_file))  # camera -> base frame (reference)
    # print(HT)
    # print("norm: ", np.linalg.norm(HT[:3, 3]))
    HT_inv = np.linalg.inv(HT)  # base frame (reference / world) -> camera
    # print(HT)
    # print("norm: ", np.linalg.norm(HT[:3, 3]))
    HT_optical = tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # camera -> optical
    HT_opengl = tf.euler_matrix(np.pi, 0.0, 0.0, axes="sxyz")  # optical -> opengl

    HT_opengl_global = HT_opengl @ HT_optical @ HT_inv  # base frame -> opengl
    # HT_opengl_global = np.linalg.inv(HT_opengl_global)
    print(HT_opengl_global)

    # HT to look at... meshrender https://github.com/BerkeleyAutomation/meshrender
    up = -HT_opengl_global[1, :3]
    center = -HT_opengl_global[2, :3]
    eye = -np.linalg.inv(HT_opengl_global[:3, :3]) @ HT_opengl_global[:3, 3]

    # ## guessed
    # center = np.array([0, 1, 0])
    # eye = np.array([0.4, -1.1, 0.4])
    # up = np.array([0, 0, 1])

    print("center: ", center.transpose())
    print("eye: ", eye.transpose())
    print("up: ", up.transpose())

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
