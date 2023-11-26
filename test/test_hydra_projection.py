import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from common import find_files

from roboreg.hydra_icp import HydraProjection
from roboreg.util import generate_o3d_robot, parse_camera_info


def test_hydra_projection() -> None:
    ############
    # parameters
    ############
    path = "test/data/high_res"
    sample_points_per_link = 1000
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ###########
    # load data
    ###########
    height, width, intrinsic_matrix = parse_camera_info(
        os.path.join(path, "left_camera_info.yaml")
    )
    masks = []
    for mask_file in find_files(path, "mask_*.png"):
        masks.append(cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE))

    robot = generate_o3d_robot(
        package_name="lbr_description",
        relative_xacro_path="urdf/med7/med7.urdf.xacro",
    )

    mesh_point_clouds = []
    for joint_state_file in find_files(path, "joint_state_*.npy"):
        joint_state = np.load(os.path.join(path, joint_state_file))
        robot.set_joint_positions(joint_state)
        mesh_point_cloud = np.concatenate(
            [
                link_point_cloud.points
                for link_point_cloud in robot.sample_point_clouds(
                    number_of_points_per_link=sample_points_per_link
                )
            ]
        )
        mesh_point_clouds.append(mesh_point_cloud)

    ##############
    # registration
    ##############
    hydra_projection = HydraProjection(
        height=height,
        width=width,
        intrinsic_matrices={"left": intrinsic_matrix},
        extrinsic_matrices={
            "left": np.eye(4),
        },
        masks={"left": masks},
        mesh_point_clouds=mesh_point_clouds,
        device=device,
    )

    print(hydra_projection._distance_maps["left"][0].shape)


if __name__ == "__main__":
    test_hydra_projection()
