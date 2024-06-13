import os
import sys
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import xacro
from ament_index_python import get_package_share_directory

from roboreg.o3d_robot import O3DRobot


def test_meshify_robot(
    package: str = "lbr_description",
    filename: str = "urdf/med7/med7.xacro",
    mappings: Dict[str, str] = {},
) -> None:
    urdf = xacro.process(
        os.path.join(get_package_share_directory(package), filename), mappings=mappings
    )

    robot = O3DRobot(urdf)
    # clouds = robot.meshes_to_point_clouds(robot.meshes)

    robot.set_joint_positions(np.array([0, 1.0, 0, 1, 0, 0, 0]))
    robot.visualize_point_clouds()
    robot.visualize_meshes()
    robot.visualize_meshes()

    robot.set_joint_positions(np.array([0, 0, 0, 0, 0, 0, 0]))
    robot.visualize_meshes()


def test_sample_points_equally() -> None:
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.xacro"
        )
    )

    robot = O3DRobot(urdf)
    clouds = robot.sample_point_clouds()
    print(clouds)
    clouds = robot.sample_point_clouds_equally()
    print(clouds)


if __name__ == "__main__":
    # test_meshify_robot("lbr_description", "urdf/med7/med7.xacro")
    test_meshify_robot("xarm_description", "urdf/xarm_device.urdf.xacro")
    # test_sample_points_equally()
