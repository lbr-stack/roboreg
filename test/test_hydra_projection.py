import os
import sys
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from roboreg.hydra_icp import HydraProjection
from roboreg.util import find_files, generate_o3d_robot, parse_camera_info


def load_data(
    path: str,
) -> Tuple[np.ndarray, int, int, np.ndarray, List[np.ndarray], List[np.ndarray], str]:
    ############
    # parameters
    ############
    path = "test/data/high_res"
    number_of_points = 200
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ###########
    # load data
    ###########
    HT = np.load(os.path.join(path, "HT_hydra_robust.npy"))
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
                for link_point_cloud in robot.sample_point_clouds_equally(
                    number_of_points=number_of_points
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
    path = "test/data/high_res"
    (
        HT,
        height,
        width,
        intrinsic_matrix,
        masks,
        mesh_point_clouds,
        device,
    ) = load_data(path)

    print("HT:\n", HT)

    ##############
    # registration
    ##############
    hydra_projection = HydraProjection(
        HT_base_cam_init=HT,
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
    HT_base_cam_optimal = hydra_projection.optimize(max_iterations=1000)
    print("HT_base_cam_optimal:\n", HT_base_cam_optimal)
    np.save(os.path.join(path, "HT_base_cam_optimal_new.npy"), HT_base_cam_optimal)


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
    idx = 0
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
        hydra_projection._mesh_point_clouds[idx],
        HT_optical_base=hydra_projection._HT_optical_base.unsqueeze(0),
        intrinsic_matrix=hydra_projection._intrinsic_matrices[key],
    )

    # normalize points
    normalized_points = hydra_projection._normalize_projected_points(projected_points)

    # plot points
    normalized_points_np = normalized_points.cpu().numpy()
    plt.scatter(normalized_points_np[0, :, 0], -1 * normalized_points_np[0, :, 1])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

    # mask samples
    mask_samples = hydra_projection._mask_grid_samples(
        normalized_points, hydra_projection._boundary_masks[key][idx].unsqueeze(0)
    )

    # plot masked points
    normalized_masked_points = normalized_points[mask_samples > 0.0]

    distance_map_samples = hydra_projection._distance_map_grid_samples(
        normalized_masked_points.unsqueeze(0),
        hydra_projection._distance_maps[key][idx].unsqueeze(0),
    )

    normalized_masked_points_np = normalized_masked_points.cpu().numpy()
    plt.scatter(
        normalized_masked_points_np[:, 0],
        -1 * normalized_masked_points_np[:, 1],
        c=distance_map_samples.squeeze().cpu().numpy(),
    )
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


if __name__ == "__main__":
    test_hydra_projection_full()
    # test_hydra_project_points()
