import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from common import load_data, visualize_registration

from roboreg.hydra_icp import hydra_centroid_alignment, hydra_robust_icp
from roboreg.ray_cast import RayCastRobot
from roboreg.util import generate_o3d_robot


def test_hydra_robust_icp():
    prefix = "test/data/low_res"
    idcs = [i for i in range(8)]
    observed_xyzs, mesh_xyzs, mesh_xyzs_normals = load_data(
        idcs=idcs,
        scan=False,
        visualize=False,
        prefix=prefix,
        number_of_points_per_link=1000,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to torch
    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = torch.from_numpy(observed_xyzs[i]).to(
            dtype=torch.float32, device=device
        )
        mesh_xyzs[i] = torch.from_numpy(mesh_xyzs[i]).to(
            dtype=torch.float32, device=device
        )
        mesh_xyzs_normals[i] = torch.from_numpy(mesh_xyzs_normals[i]).to(
            dtype=torch.float32, device=device
        )

    HT_init = hydra_centroid_alignment(observed_xyzs, mesh_xyzs)
    HT = hydra_robust_icp(
        HT_init,
        observed_xyzs,
        mesh_xyzs,
        mesh_xyzs_normals,
        max_distance=0.2,
        outer_max_iter=int(30),
        inner_max_iter=10,
    )

    # cast HT
    robot = generate_o3d_robot()
    joint_states = [np.load(f"{prefix}/joint_state_{i}.npy") for i in idcs]
    cast = RayCastRobot(robot)

    # to optical frame
    max_distance = 0.1
    for iter in range(3):
        import transformations as tf

        HT = HT.cpu().numpy()
        HT_cast = HT @ tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # to optical frame
        HT_cast = np.linalg.inv(HT_cast)

        up = -HT_cast[1, :3]
        eye = -np.linalg.inv(HT_cast[:3, :3]) @ HT_cast[:3, 3]
        center = eye + HT_cast[2, :3]

        mesh_xyzs = []
        mesh_xyzs_normals = []

        for idx, joint_state in enumerate(joint_states):
            cast.robot.set_joint_positions(joint_state)
            pcd = cast.cast(
                fov_deg=120,
                center=center,
                eye=eye,
                up=up,
                width_px=1280,
                height_px=960,
            )
            try:
                mesh_xyzs.append(
                    torch.from_numpy(pcd.point.positions.numpy()).to(device).float()
                )
                print(mesh_xyzs[-1].shape)
                pcd.estimate_normals()
                mesh_xyzs_normals.append(
                    torch.from_numpy(pcd.point.normals.numpy()).to(device).float()
                )
            except:
                print("Failed to cast.")
                mesh_xyzs.pop(idx)
                observed_xyzs.pop(idx)
                continue

        # re-run hydra robust icp
        HT = torch.from_numpy(HT).to(device).float()
        HT = hydra_robust_icp(
            HT,
            observed_xyzs,
            mesh_xyzs,
            mesh_xyzs_normals,
            max_distance=max_distance / 10.0**iter,
            outer_max_iter=int(30),
            inner_max_iter=10,
        )

    # to numpy
    HT = HT.cpu().numpy()
    np.save(os.path.join(prefix, "HT_hydra_robust.npy"), HT)

    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = observed_xyzs[i].cpu().numpy()
        mesh_xyzs[i] = mesh_xyzs[i].cpu().numpy()

    visualize_registration(observed_xyzs, mesh_xyzs, np.linalg.inv(HT))


if __name__ == "__main__":
    test_hydra_robust_icp()
