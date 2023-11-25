import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from common import find_files

from roboreg.hydra_icp import HydraProjection
from roboreg.util import generate_o3d_robot


def test_hydra_projection() -> None:
    ## parameters
    path = "test/data/high_res"
    sample_points_per_link = 1000
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    height = 540
    width = 960
    intrinsic_matrix = np.array(
        [
            [533.9981079101562, 0.0, 478.0845642089844],
            [0.0, 533.9981079101562, 260.9956970214844],
            [0.0, 0.0, 1.0],
        ]
    )

    ## load data
    masks = []
    for mask_file in find_files(path, "mask_*.png"):
        masks.append(cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE))

    robot = generate_o3d_robot(
        package_name="lbr_description",
        relative_xacro_path="urdf/med7/med7.urdf.xacro",
    )

    joint_states = []
    for joint_state_file in find_files(path, "joint_states_*.npy"):
        joint_states.append(np.load(os.path.join(path, joint_state_file)))

    # TODO: sample pcds and shape -> [N, M, 3]
    pcds = robot.sample_point_clouds(number_of_points_per_link=sample_points_per_link)

    ## registration
    hydra_projection = HydraProjection(
        height=height,
        width=width,
        intrinsic_matrices={"left": intrinsic_matrix},
        extrinsic_matrices={
            "left": np.eye(4),
        },
        masks={"left": masks},
        # mesh_point_clouds=
        device=device,
    )

    print(hydra_projection._distance_maps["left"][0].shape)


if __name__ == "__main__":
    test_hydra_projection()
