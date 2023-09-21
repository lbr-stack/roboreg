import os
import cv2
import numpy as np
import pyvista as pv
import xacro

from ament_index_python import get_package_share_directory

from roboreg.meshify_robot import MeshifyRobot
from roboreg.register import RobustICPRegister, ICPRegister, clean_xyz


def test_regsiter() -> None:
    # load data
    mask = cv2.imread("test/data/mask_3.png", cv2.IMREAD_GRAYSCALE)
    observed_xyz = np.load("test/data/xyz_3.npy")
    joint_state = np.load("test/data/joint_state_3.npy")

    # clean cloud
    clean_observed_xyz = clean_xyz(observed_xyz, mask)

    # visualize clean cloud
    plotter = pv.Plotter()
    plotter.background_color = "black"
    plotter.add_mesh(clean_observed_xyz, point_size=2.0, color="white")
    plotter.show()

    # load mesh
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )

    # transform mesh
    robot = MeshifyRobot(urdf=urdf, resolution="collision")
    meshes = robot.transformed_meshes(joint_state)
    mesh_xyz = robot.meshes_to_point_cloud(meshes)
    mesh_xyz = robot.homogenous_point_cloud_sampling(mesh_xyz, 3000)

    # register ICP
    register = ICPRegister(observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz)
    register.register()
    register.draw_registration_result()

    # register RobustICP
    register = RobustICPRegister(observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz)
    register.register()
    register.draw_registration_result()


if __name__ == "__main__":
    test_regsiter()
