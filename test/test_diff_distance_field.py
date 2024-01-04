import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import transformations as tf

from roboreg.util import generate_o3d_robot, parse_camera_info


def main() -> None:
    path = "test/data/high_res"

    # load data
    height, width, intrinsic_matrix = parse_camera_info(
        os.path.join(path, "left_camera_info.yaml")
    )

    HT_base_cam = np.load(os.path.join(path, "HT_hydra_robust.npy"))
    HT_cam_optical = tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # camera -> optical
    HT_base_optical = HT_base_cam @ HT_cam_optical  # base frame -> optical
    HT_optical_base = np.linalg.inv(HT_base_optical)
    robot = generate_o3d_robot()
    mesh_point_cloud = [
        np.array(link_point_cloud.points)
        for link_point_cloud in robot.sample_point_clouds(
            number_of_points_per_link=1000
        )
    ]
    mesh_point_cloud = np.concatenate(mesh_point_cloud, axis=0)

    # to torch
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix)
    HT_optical_base = torch.from_numpy(HT_optical_base)
    mesh_point_cloud = torch.from_numpy(mesh_point_cloud)

    # project points
    mesh_projected = torch.matmul(
        mesh_point_cloud @ HT_optical_base[:3, :3].T + HT_optical_base[:3, 3],
        intrinsic_matrix.T,
    )

    # normalize
    mesh_projected = mesh_projected / mesh_projected[..., 2].unsqueeze(-1)
    mesh_projected = mesh_projected[..., :2]

    # remove beyond height width
    mask = torch.logical_and(
        torch.logical_and(mesh_projected[..., 0] >= 0, mesh_projected[..., 0] < width),
        torch.logical_and(mesh_projected[..., 1] >= 0, mesh_projected[..., 1] < height),
    )
    mesh_projected = mesh_projected[mask]

    zeros = torch.zeros([height, width])
    zeros[mesh_projected[:, 1].int(), mesh_projected[:, 0].int()] = 1.0

    # zeros = gaussian_blur(zeros.unsqueeze(0), kernel_size=10, sigma=1.0).squeeze()

    # # zeros = torch.scatter_reduce(
    # #     zeros, dim=-1, index=mesh_projected[:, 0].int(), reduce="max"
    # # )
    zeros = zeros.numpy()

    import cv2

    cv2.imshow("zeros", zeros)
    cv2.waitKey(0)

    # # to numpy anp plot
    # mesh_projected = mesh_projected.numpy()
    # import matplotlib.pyplot as plt

    # plt.scatter(mesh_projected[:, 0], -1 * mesh_projected[:, 1])
    # plt.xlim(0, width)
    # plt.ylim(-height, 0)
    # plt.show()

    # render
    # ones = torch.ones_


if __name__ == "__main__":
    main()
