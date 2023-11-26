import os
import sys
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from common import find_files
import matplotlib.pyplot as plt

from roboreg.hydra_icp import HydraProjection
from roboreg.util import generate_o3d_robot, parse_camera_info


def load_data(
    path: str,
) -> Tuple[np.ndarray, int, int, np.ndarray, List[np.ndarray], List[np.ndarray], str]:
    ############
    # parameters
    ############
    path = "test/data/high_res"
    sample_points_per_link = 1000
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ###########
    # load data
    ###########
    HT = np.load(os.path.join(path, "HT_hydra_robust.npy"))
    height, width, intrinsic_matrix = parse_camera_info(
        os.path.join(path, "left_camera_info.yaml")
    )
    masks = []
    for mask_file in find_files(path, "mask_*.png")[:1]:  # TODO: replace this by all!
        masks.append(cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE))

    robot = generate_o3d_robot(
        package_name="lbr_description",
        relative_xacro_path="urdf/med7/med7.urdf.xacro",
    )

    mesh_point_clouds = []
    for joint_state_file in find_files(path, "joint_state_*.npy")[
        :1
    ]:  # TODO: replace this by all!
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

    return (
        HT,
        height,
        width,
        intrinsic_matrix,
        masks,
        mesh_point_clouds,
        device,
    )


def test_hydra_projection_full() -> None:
    (
        HT,
        height,
        width,
        intrinsic_matrix,
        masks,
        mesh_point_clouds,
        device,
    ) = load_data("test/data/high_res")

    ##############
    # registration
    ##############
    hydra_projection = HydraProjection(
        HT_init=HT,
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


def test_hydra_project_points() -> None:
    (
        HT,
        height,
        width,
        intrinsic_matrix,
        masks,
        mesh_point_clouds,
        device,
    ) = load_data("test/data/high_res")

    ##################
    # hydra projection
    ##################
    key = "left"
    hydra_projection = HydraProjection(
        HT_base_cam_init=HT,
        height=height,
        width=width,
        intrinsic_matrices={key: intrinsic_matrix},
        extrinsic_matrices={
            key: np.eye(4),
        },
        masks={key: masks},
        mesh_point_clouds=mesh_point_clouds,
        device=device,
    )

    # project points
    projected_points = hydra_projection._project_points(
        hydra_projection._mesh_point_clouds[0],
        HT_optical_base=hydra_projection._HT_optical_base.unsqueeze(0),
        intrinsic_matrix=hydra_projection._intrinsic_matrices[key],
    )

    # normalize points
    normalized_points = hydra_projection._normalize_projected_points(projected_points)

    # plot points
    normalized_points = normalized_points.cpu().numpy()
    plt.scatter(normalized_points[0, :, 0], -1 * normalized_points[0, :, 1])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


if __name__ == "__main__":
    # test_hydra_projection_full()
    test_hydra_project_points()
