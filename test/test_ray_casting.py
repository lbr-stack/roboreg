import cv2
import numpy as np
import open3d as o3d
import torch
import transformations as tf

from roboreg.ray_cast import RayCastRobot
from roboreg.util import generate_o3d_robot, clean_xyz
from roboreg.hydra_icp import hydra_centroid_alignment


def test_ray_casting() -> None:
    robot = generate_o3d_robot()
    robot.set_joint_positions(np.array([0, 1, 0, 1, 0, 0, 0]))

    cast = RayCastRobot(robot)

    pcd = cast.cast(
        fov_deg=90,
        center=[0, 0, 0],
        eye=[0, 2, 0],
        up=[0, 0, 1],
        width_px=640,
        height_px=480,
    )
    o3d.visualization.draw_geometries(
        [pcd.to_legacy()],
    )

    cast.robot.set_joint_positions(np.array([0, 1, 1, -1, 0, 0, 0.5]))
    pcd = cast.cast(
        fov_deg=90,
        center=[0, 0, 0],
        eye=[0, -2, 0],
        up=[0, 0, 1],
        width_px=640,
        height_px=480,
    )
    o3d.visualization.draw_geometries(
        [pcd.to_legacy()],
    )

    # tensor documentation: http://www.open3d.org/docs/latest/tutorial/Basic/tensor.html
    # add noise
    points = pcd.point.positions.numpy()
    points += np.random.normal(0, 0.01, points.shape)

    pcd_new = o3d.t.geometry.PointCloud()
    pcd_new.point.points = o3d.core.Tensor.from_numpy(points)  # shared memory

    o3d.visualization.draw_geometries(
        [pcd.to_legacy()],
    )


def test_ray_cast_homogeneous() -> None:
    robot = generate_o3d_robot()
    robot.set_joint_positions(np.array([0, 1, 0, 1, 0, 0, 0]))

    # HT = np.eye(4)
    HT = tf.euler_matrix(0.0, -np.pi / 2.0, 0.0, axes="sxyz")
    HT[:3, 3] = np.array([3.0, 0.0, 0.0])

    cast = RayCastRobot(robot)

    intrinsic_matrix = np.eye(3)
    intrinsic_matrix[0, 0] = 640  # some values
    intrinsic_matrix[1, 1] = 480

    pcd = cast.cast_ht(
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=np.linalg.inv(HT),
        width_px=640,
        height_px=480,
    )
    o3d.visualization.draw_geometries(
        [pcd.to_legacy()],
    )


def test_ray_cast_with_centroid_align() -> None:
    prefix = "test/data/low_res"
    observed_xyzs = []
    mesh_xyzs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load robot
    robot = generate_o3d_robot()

    for idx in [0, 1, 2]:
        # load data
        mask = cv2.imread(f"{prefix}/mask_{idx}.png", cv2.IMREAD_GRAYSCALE)
        observed_xyz = np.load(f"{prefix}/xyz_{idx}.npy")
        joint_state = np.load(f"{prefix}/joint_state_{idx}.npy")

        # clean cloud
        observed_xyzs.append(
            torch.from_numpy(clean_xyz(observed_xyz, mask)).to(device).double()
        )

        robot.set_joint_positions(joint_state)
        pcds = robot.sample_point_clouds()
        mesh_xyzs.append(
            torch.from_numpy(np.concatenate([np.array(pcd.points) for pcd in pcds]))
            .to(device)
            .double()
        )

    HT = (
        hydra_centroid_alignment(
            Xs=observed_xyzs,
            Ys=mesh_xyzs,
        )
        .cpu()
        .numpy()
    )

    # to optical frame
    HT = HT @ tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # to optical frame
    HT = np.linalg.inv(HT)

    # given HT, render
    cast = RayCastRobot(robot)

    width = 640
    height = 360
    intrinsic_matrix = np.array(
        [
            [263.8703308105469, 0.0, 318.25634765625],
            [0.0, 263.8703308105469, 174.2410888671875],
            [0.0, 0.0, 1.0],
        ]
    )

    pcd = cast.cast_ht(
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=HT,
        width_px=width,
        height_px=height,
    )

    o3d.visualization.draw_geometries(
        [pcd.to_legacy()],
    )


if __name__ == "__main__":
    # test_ray_cast_homogeneous()
    # test_ray_casting()
    test_ray_cast_with_centroid_align()
