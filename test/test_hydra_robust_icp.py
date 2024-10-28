import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from roboreg.differentiable import TorchKinematics, TorchMeshContainer
from roboreg.hydra_icp import hydra_centroid_alignment, hydra_robust_icp
from roboreg.io import URDFParser, parse_hydra_data
from roboreg.util import (
    RegistrationVisualizer,
    clean_xyz,
    compute_vertex_normals,
    from_homogeneous,
    mask_boundary,
)


def test_hydra_robust_icp() -> None:
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
    )

    # instantiate kinematics
    urder_parser = URDFParser()
    urder_parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)
    kinematics = TorchKinematics(
        urder_parser=urder_parser,
        device=device,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
    )

    # instantiate mesh
    batch_size = len(joint_states)
    meshes = TorchMeshContainer(
        mesh_paths=urder_parser.ros_package_mesh_paths(
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

    # mesh vertices to list
    mesh_vertices = from_homogeneous(mesh_vertices)
    mesh_vertices = [mesh_vertices[i].contiguous() for i in range(batch_size)]
    mesh_normals = []
    for i in range(batch_size):
        mesh_normals.append(
            compute_vertex_normals(vertices=mesh_vertices[i], faces=meshes.faces)
        )

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
        mesh_normals[i] = mesh_normals[i][idx]

    HT_init = hydra_centroid_alignment(observed_vertices, mesh_vertices)
    HT = hydra_robust_icp(
        HT_init,
        observed_vertices,
        mesh_vertices,
        mesh_normals,
        max_distance=0.01,
        outer_max_iter=int(50),
        inner_max_iter=10,
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
    HT = HT.cpu().numpy()
    np.save(os.path.join(path, "HT_hydra_robust.npy"), HT)


if __name__ == "__main__":
    test_hydra_robust_icp()
