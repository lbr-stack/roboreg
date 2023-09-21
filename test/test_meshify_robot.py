import os

import numpy as np
import xacro
from ament_index_python import get_package_share_directory

from roboreg.meshify_robot import MeshifyRobot


def test_meshify_robot() -> None:
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )

    meshify_robot = MeshifyRobot(urdf, "collision")

    for _ in range(2):
        q = np.random.uniform(-np.pi / 2, np.pi / 2, meshify_robot.dof)
        meshes = meshify_robot.transformed_meshes(q)
        meshify_robot.plot_meshes(meshes)
        meshify_robot.plot_point_clouds(meshes)


if __name__ == "__main__":
    test_meshify_robot()
