import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import transformations as tf

from roboreg.differentiable import TorchKinematics, TorchMeshContainer
from roboreg.hydra_icp import hydra_centroid_alignment, hydra_icp
from roboreg.io import URDFParser, parse_hydra_data
from roboreg.util.mask import mask_boundary
from roboreg.util.points import clean_xyz, from_homogeneous
from roboreg.util.viz import RegistrationVisualizer


def test_hydra_centroid_alignment():
    mesh_centroids = [
        torch.FloatTensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.FloatTensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.FloatTensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    ]

    HT_random = torch.from_numpy(tf.random_rotation_matrix()).float()
    HT_random[:3, 3] = torch.FloatTensor([1.0, 2.0, 3.0])

    observed_centroids = [
        mesh_centroid @ HT_random[:3, :3].T + HT_random[:3, 3]
        for mesh_centroid in mesh_centroids
    ]

    HT = hydra_centroid_alignment(mesh_centroids, observed_centroids)

    assert torch.allclose(HT, HT_random)


def test_hydra_icp():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ros_package = "lbr_description"
    xacro_path = "urdf/med7/med7.xacro"
    root_link_name = "lbr_link_0"
    end_link_name = "lbr_link_7"
    path = "test/data/lbr_med7/zed2i/high_res"
    joint_states_pattern = "joint_states_*.npy"
    mask_pattern = "mask*.png"
    xyz_pattern = "xyz_*.npy"

    # load data
    joint_states, masks, xyzs = parse_hydra_data(
        path=path,
        joint_states_pattern=joint_states_pattern,
        mask_pattern=mask_pattern,
        xyz_pattern=xyz_pattern,
        device=device,
    )

    # instantiate kinematics
    parser = URDFParser()
    parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)
    kinematics = TorchKinematics(
        parser.urdf,
        device=device,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
    )

    # instantiate mesh
    batch_size = len(joint_states)
    meshes = TorchMeshContainer(
        mesh_paths=parser.ros_package_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name
        ),
        batch_size=batch_size,
        device=device,
    )

    # process data
    mesh_vertices = meshes.vertices.clone()
    joint_states = torch.tensor(
        np.array(joint_states), dtype=torch.float32, device=device
    )
    ht_lookup = kinematics.mesh_forward_kinematics(joint_states)
    for link_name, ht in ht_lookup.items():
        mesh_vertices[
            :,
            meshes.lower_vertex_index_lookup[
                link_name
            ] : meshes.upper_vertex_index_lookup[link_name],
        ] = torch.matmul(
            mesh_vertices[
                :,
                meshes.lower_vertex_index_lookup[
                    link_name
                ] : meshes.upper_vertex_index_lookup[link_name],
            ],
            ht.transpose(-1, -2),
        )
    mesh_vertices = from_homogeneous(mesh_vertices)

    # mesh vertices to list
    mesh_vertices = [mesh_vertices[i].contiguous() for i in range(batch_size)]

    # clean observed vertices and turn into tensor
    observed_vertices = [
        torch.tensor(
            clean_xyz(xyz=xyz, mask=mask_boundary(mask)),
            dtype=torch.float32,
            device=device,
        )
        for xyz, mask in zip(xyzs, masks)
    ]

    # sample 5000 points per mesh
    for i in range(batch_size):
        idx = torch.randperm(mesh_vertices[i].shape[0])[:5000]
        mesh_vertices[i] = mesh_vertices[i][idx]

    HT_init = hydra_centroid_alignment(observed_vertices, mesh_vertices)
    HT = hydra_icp(
        HT_init,
        observed_vertices,
        mesh_vertices,
        max_distance=0.1,
        max_iter=int(1e3),
        rmse_change=1e-8,
    )

    # visualize
    visualizer = RegistrationVisualizer()
    visualizer(mesh_vertices=mesh_vertices, observed_vertices=observed_vertices)
    visualizer(
        mesh_vertices=mesh_vertices,
        observed_vertices=observed_vertices,
        HT=torch.linalg.inv(HT),
    )

    # to numpy
    np.save(os.path.join(path, "HT_hydra.npy"), HT.cpu().numpy())


if __name__ == "__main__":
    # test_hydra_centroid_alignment()
    test_hydra_icp()
