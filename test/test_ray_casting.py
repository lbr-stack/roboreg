import os

import numpy as np
import open3d as o3d
import xacro
from ament_index_python import get_package_share_directory

from roboreg.o3d_robot import O3DRobot
from roboreg.ray_cast import RayCastRobot


def test_ray_casting() -> None:
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )

    robot = O3DRobot(urdf)
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


if __name__ == "__main__":
    test_ray_casting()
