import numpy as np
import open3d as o3d
import transformations as tf

from roboreg.ray_cast import RayCastRobot
from roboreg.util import generate_o3d_robot


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

    intrinsic_matrix = np.array(
        [
            [533.9981079101562, 0.0, 478.0845642089844],
            [0.0, 533.9981079101562, 260.9956970214844],
            [0.0, 0.0, 1.0],
        ]
    )

    pcd = cast.cast_ht(
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=np.linalg.inv(HT),
        width_px=640,
        height_px=480,
    )
    o3d.visualization.draw_geometries(
        [pcd.to_legacy()],
    )


if __name__ == "__main__":
    test_ray_cast_homogeneous()
    # test_ray_casting()
