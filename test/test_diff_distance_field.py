import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import transformations as tf
from torchvision.transforms.functional import gaussian_blur

from roboreg.util import generate_o3d_robot, parse_camera_info


def main() -> None:
    path = "test/data/high_res"

    # load data
    height, width, intrinsic_matrix = parse_camera_info(
        os.path.join(path, "left_camera_info.yaml")
    )

    idx = 1

    mask = cv2.imread(os.path.join(path, f"mask_{idx}.png"), cv2.IMREAD_GRAYSCALE)
    joint_states = np.load(os.path.join(path, f"joint_state_{idx}.npy"))

    HT_base_cam = np.load(os.path.join(path, "HT_hydra_robust.npy"))
    HT_cam_optical = tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # camera -> optical
    HT_base_optical = HT_base_cam @ HT_cam_optical  # base frame -> optical
    HT_optical_base = np.linalg.inv(HT_base_optical)
    robot = generate_o3d_robot()
    robot.set_joint_positions(joint_states)
    mesh_point_cloud = [
        np.array(link_point_cloud.points)
        for link_point_cloud in robot.sample_point_clouds_equally(number_of_points=5000)
    ]
    mesh_point_cloud = np.concatenate(mesh_point_cloud, axis=0)

    # to torch
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix)
    HT_optical_base = torch.from_numpy(HT_optical_base)
    mesh_point_cloud = torch.from_numpy(mesh_point_cloud)

    ## dummy rendering pipeline
    # project points
    mesh_projected = torch.matmul(
        mesh_point_cloud @ HT_optical_base[:3, :3].T + HT_optical_base[:3, 3],
        intrinsic_matrix.T,
    )

    # normalize
    mesh_projected = mesh_projected / mesh_projected[..., 2].unsqueeze(-1)
    mesh_projected = mesh_projected[..., :2]

    # remove beyond height width
    mask_mesh = torch.logical_and(
        torch.logical_and(mesh_projected[..., 0] >= 0, mesh_projected[..., 0] < width),
        torch.logical_and(mesh_projected[..., 1] >= 0, mesh_projected[..., 1] < height),
    )
    mesh_projected = mesh_projected[mask_mesh]

    naiv_render = torch.zeros([height, width])
    naiv_render[mesh_projected[:, 1].int(), mesh_projected[:, 0].int()] = 1.0

    naiv_render = gaussian_blur(
        naiv_render.unsqueeze(0), kernel_size=9, sigma=10.0
    ).squeeze()

    # where greater threshold set one
    # naiv_render[naiv_render > 0.01] = 1.0

    naiv_render = naiv_render.numpy()
    cv2.imshow("naiv_render", naiv_render)
    cv2.imshow("mask", mask)
    cv2.imshow("mask - naiv_render", naiv_render - mask.astype(np.float) / 255.0)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
