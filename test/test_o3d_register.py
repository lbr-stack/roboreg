import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common import load_data

from roboreg.o3d_register import GlobalRegistration, ICPRegister, RobustICPRegister
from roboreg.util import generate_o3d_robot


def test_o3d_icp_register(
    clean_observed_xyz: np.ndarray, mesh_xyz: np.ndarray, visualize: bool = True
) -> np.ndarray:
    # register ICP
    icp_register = ICPRegister(observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz)
    icp_register.register()
    icp_register._trans_init = icp_register._transformation
    icp_register.register(0.05)
    icp_register._trans_init = icp_register._transformation
    icp_register.register(0.01)
    if visualize:
        icp_register.draw_registration_result()
    return icp_register._transformation


def test_o3d_icp_raycast_register(
    clean_observed_xyz: np.ndarray, mesh_xyz: np.ndarray, visualize: bool = True
) -> np.ndarray:
    # register ICP
    icp_register = ICPRegister(observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz)
    icp_register.register()
    icp_register._trans_init = icp_register._transformation
    icp_register.register(0.05)
    icp_register._trans_init = icp_register._transformation
    icp_register.register(0.01)

    # raycast
    from roboreg.o3d_robot import O3DRobot
    from roboreg.ray_cast import RayCastRobot

    joint_state = np.load(f"test/data/high_res/joint_state_{idx}.npy")

    # load robot
    robot = generate_o3d_robot()

    raycast = RayCastRobot(robot)
    raycast.robot.set_joint_positions(joint_state)
    pcd = raycast.cast(
        fov_deg=90,
        center=[0.0, 0.0, 0.5],
        eye=icp_register._transformation[:3, 3],
        up=[0.0, 0.0, 1.0],
        width_px=640,
        height_px=480,
    )
    import open3d as o3d

    o3d.visualization.draw_geometries([pcd.to_legacy()])

    # re-run ICP
    icp_register.mesh_xyz_pcd = pcd.point.positions.numpy()
    icp_register.register(0.01)

    if visualize:
        icp_register.draw_registration_result()
    return icp_register._transformation


def test_o3d_robust_icp_register(
    clean_observed_xyz: np.ndarray, mesh_xyz: np.ndarray, visualize: bool = True
) -> np.ndarray:
    # register RobustICP
    robust_icp_register = RobustICPRegister(
        observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz
    )
    robust_icp_register.register()
    if visualize:
        robust_icp_register.draw_registration_result()
    return robust_icp_register._transformation


def test_o3d_global_register(
    clean_observed_xyz: np.ndarray, mesh_xyz: np.ndarray, visualize: bool = True
) -> np.ndarray:
    # register Global
    global_register = GlobalRegistration(
        observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz
    )
    global_register.register(voxel_size=0.03)
    if visualize:
        global_register.draw_registration_result()

    # refine registration using ICP
    icp_global_register = ICPRegister(
        observed_xyz=clean_observed_xyz, mesh_xyz=mesh_xyz
    )
    icp_global_register._trans_init = global_register._transformation
    icp_global_register.register(threshold=0.1)
    if visualize:
        icp_global_register.draw_registration_result()
    return icp_global_register._transformation


if __name__ == "__main__":
    icp_tranformations = []
    icp_raycast_tranformations = []
    for idx in range(17):
        print(f"Processing {idx}")
        N = 5
        clean_observed_xyzs, mesh_xyzs, _ = load_data(
            idcs=[idx + i for i in range(N)], visualize=False
        )
        # concatenate
        clean_observed_xyz = np.concatenate(clean_observed_xyzs, axis=0)
        mesh_xyz = np.concatenate(mesh_xyzs, axis=0)
        # discard_list = [1, 6, 9, 10, 11, 12, 14, 16]
        # if idx in discard_list:  # garbage registration
        #     continue
        # icp_tranformation = test_o3d_icp_register(
        #     clean_observed_xyz=clean_observed_xyz,
        #     mesh_xyz=mesh_xyz,
        #     visualize=True,
        # )
        # icp_tranformations.append(icp_tranformation)
        # icp_raycast_tranformation = test_o3d_icp_raycast_register(
        #     clean_observed_xyz=clean_observed_xyz,
        #     mesh_xyz=mesh_xyz,
        #     visualize=True,
        # )
        # icp_raycast_tranformations.append(icp_raycast_tranformation)
        # robust_icp_tranformation = test_o3d_robust_icp_register(
        #     clean_observed_xyz=clean_observed_xyz,
        #     mesh_xyz=mesh_xyz,
        #     visualize=True,
        # )

        global_tranformation = test_o3d_global_register(
            clean_observed_xyz=clean_observed_xyz,
            mesh_xyz=mesh_xyz,
            visualize=True,
        )

    # mean
    np_icp_tranformations = np.array(icp_tranformations)
    mean = np.mean(np_icp_tranformations, axis=0)
    std = np.std(np_icp_tranformations, axis=0)

    import transformations

    eulers = []
    for icp_tranformation in icp_tranformations:
        eulers.append(transformations.euler_from_matrix(icp_tranformation))
    eulers = np.array(eulers)
    euler_mean = np.mean(eulers, axis=0)
    euler_std = np.std(eulers, axis=0)

    print("translation mean: ", mean[:3, 3])
    print("translation std:  ", std[:3, 3])
    print("rotation mean: ", euler_mean * 180 / np.pi)
    print("rotation std:  ", euler_std * 180 / np.pi)
