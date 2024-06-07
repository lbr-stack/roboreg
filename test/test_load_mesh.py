import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d

from roboreg.o3d_robot import O3DRobot


def test_load_mesh(path: str) -> None:
    mesh = O3DRobot._load_mesh(path)

    # visualize
    o3d.visualization.draw([mesh])


if __name__ == "__main__":
    test_load_mesh("test/data/xarm/mesh/link_base.STL")
    # test_load_mesh("test/data/lbr_med7/mesh/link_0.stl")
