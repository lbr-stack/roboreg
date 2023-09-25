import os

import cv2
import numpy as np
import pyvista as pv
import xacro
from ament_index_python import get_package_share_directory

from roboreg.meshify_robot import MeshifyRobot
from roboreg.o3d_register import GlobalRegistration, ICPRegister, RobustICPRegister
from roboreg.util import clean_xyz


def test_o3d_register(idx: int = 1) -> None:
    # load data
    mask = cv2.imread(f"test/data/mask_{idx}.png", cv2.IMREAD_GRAYSCALE)
    observed_xyz = np.load(f"test/data/xyz_{idx}.npy")
    joint_state = np.load(f"test/data/joint_state_{idx}.npy")

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
    # mesh_xyz = robot.homogenous_point_cloud_sampling(mesh_xyz, 3000)

    # register ICP
    icp_register = ICPRegister(observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz)
    icp_register.register()
    icp_register.draw_registration_result()

    # register RobustICP
    robust_icp_register = RobustICPRegister(
        observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz
    )
    robust_icp_register.register()
    robust_icp_register.draw_registration_result()

    # register Global
    global_register = GlobalRegistration(
        observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz
    )
    global_register.register(voxel_size=0.02)
    global_register.draw_registration_result()

    # refine registration using ICP
    icp_register = ICPRegister(observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz)
    icp_register._trans_init = global_register._transformation
    icp_register.register()
    icp_register.draw_registration_result()


if __name__ == "__main__":
    test_o3d_register(idx=1)
