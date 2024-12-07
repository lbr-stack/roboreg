import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import transformations as tf

from roboreg.differentiable import TorchKinematics, TorchMeshContainer
from roboreg.hydra_icp import (
    hydra_centroid_alignment,
    hydra_correspondence_indices,
    hydra_icp,
    hydra_robust_icp,
)
from roboreg.io import URDFParser, parse_camera_info, parse_hydra_data
from roboreg.util import (
    RegistrationVisualizer,
    clean_xyz,
    compute_vertex_normals,
    depth_to_xyz,
    from_homogeneous,
    generate_ht_optical,
    mask_extract_boundary,
    to_homogeneous,
)


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


def test_hydra_correspondence_indices() -> None:
    M = 100
    N = 10
    dim = 3

    def test_index_shape(
        indices: torch.Tensor,
        mask: torch.Tensor,
        target_shape: torch.Size,
        max_val: int,
        min_val: int = 0,
    ) -> None:
        if indices.shape != mask.shape:
            raise ValueError("Indices and mask shapes do not match.")
        if indices.shape != target_shape:
            raise ValueError("Indices shape is incorrect.")
        if indices.max() >= max_val:
            raise ValueError("Indices contain out of bounds indices.")
        if indices.min() < min_val:
            raise ValueError("Indices contain negative indices.")

    # single input
    input = torch.rand(M, dim)
    target = torch.rand(N, dim)  # e.g. the mesh vertices
    matchindices, mask = hydra_correspondence_indices(
        input, target, max_distance=np.sqrt(dim) / 2.0  # remove some elements randomly
    )
    test_index_shape(matchindices, mask, torch.Size([M]), N)

    # batched input
    batch_size = 2
    input = torch.rand(batch_size, M, dim)
    target = torch.rand(batch_size, N, dim)
    matchindices, mask = hydra_correspondence_indices(
        input, target, max_distance=np.sqrt(dim) / 2.0
    )
    test_index_shape(matchindices, mask, torch.Size([batch_size, M]), N)

    # test for inverted case
    M = 10
    N = 100

    input = torch.rand(M, dim)
    target = torch.rand(N, dim)
    matchindices, mask = hydra_correspondence_indices(
        input, target, max_distance=np.sqrt(dim) / 2.0
    )
    test_index_shape(matchindices, mask, torch.Size([M]), N)


def test_hydra_icp():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ros_package = "lbr_description"
    xacro_path = "urdf/med7/med7.xacro"
    root_link_name = "lbr_link_0"
    end_link_name = "lbr_link_7"
    path = "test/data/lbr_med7/zed2i"
    camera_info_file = "left_camera_info.yaml"
    joint_states_pattern = "joint_states_*.npy"
    mask_pattern = "mask_sam2_left_*.png"
    depth_pattern = "depth_*.npy"

    # load data
    joint_states, masks, depths = parse_hydra_data(
        path=path,
        joint_states_pattern=joint_states_pattern,
        mask_pattern=mask_pattern,
        depth_pattern=depth_pattern,
    )
    height, width, intrinsics = parse_camera_info(
        camera_info_file=os.path.join(path, camera_info_file)
    )

    # instantiate kinematics
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)
    kinematics = TorchKinematics(
        urdf_parser=urdf_parser,
        device=device,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
    )

    # instantiate mesh
    batch_size = len(joint_states)
    meshes = TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name
        ),
        batch_size=batch_size,
        device=device,
    )

    # perform forward kinematics
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

    # turn depths into xyzs
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    depths = torch.tensor(np.array(depths), dtype=torch.float32, device=device)
    xyzs = depth_to_xyz(depth=depths, intrinsics=intrinsics, z_max=1.5)

    # flatten BxHxWx3 -> Bx(H*W)x3
    xyzs = xyzs.view(-1, height * width, 3)
    xyzs = to_homogeneous(xyzs)
    ht_optical = generate_ht_optical(xyzs.shape[0], dtype=torch.float32, device=device)
    xyzs = torch.matmul(xyzs, ht_optical.transpose(-1, -2))
    xyzs = from_homogeneous(xyzs)

    # unflatten
    xyzs = xyzs.view(-1, height, width, 3)
    xyzs = [xyz.squeeze() for xyz in xyzs.cpu().numpy()]

    # mesh vertices to list
    mesh_vertices = [mesh_vertices[i].contiguous() for i in range(batch_size)]

    # clean observed vertices and turn into tensor
    observed_vertices = [
        torch.tensor(
            clean_xyz(xyz=xyz, mask=mask_extract_boundary(mask)),
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


def test_hydra_robust_icp() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ros_package = "lbr_description"
    xacro_path = "urdf/med7/med7.xacro"
    root_link_name = "lbr_link_0"
    end_link_name = "lbr_link_7"
    path = "test/data/lbr_med7/zed2i"
    camera_info_file = "left_camera_info.yaml"
    joint_states_pattern = "joint_states_*.npy"
    mask_pattern = "mask_sam2_left_*.png"
    depth_pattern = "depth_*.npy"

    # load data
    joint_states, masks, depths = parse_hydra_data(
        path=path,
        joint_states_pattern=joint_states_pattern,
        mask_pattern=mask_pattern,
        depth_pattern=depth_pattern,
    )
    height, width, intrinsics = parse_camera_info(
        camera_info_file=os.path.join(path, camera_info_file)
    )

    # instantiate kinematics
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)
    kinematics = TorchKinematics(
        urdf_parser=urdf_parser,
        device=device,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
    )

    # instantiate mesh
    batch_size = len(joint_states)
    meshes = TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name
        ),
        batch_size=batch_size,
        device=device,
    )

    # perform forward kinematics
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

    # turn depths into xyzs
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    depths = torch.tensor(np.array(depths), dtype=torch.float32, device=device)
    xyzs = depth_to_xyz(depth=depths, intrinsics=intrinsics, z_max=1.5)

    # flatten BxHxWx3 -> Bx(H*W)x3
    xyzs = xyzs.view(-1, height * width, 3)
    xyzs = to_homogeneous(xyzs)
    ht_optical = generate_ht_optical(xyzs.shape[0], dtype=torch.float32, device=device)
    xyzs = torch.matmul(xyzs, ht_optical.transpose(-1, -2))
    xyzs = from_homogeneous(xyzs)

    # unflatten
    xyzs = xyzs.view(-1, height, width, 3)
    xyzs = [xyz.squeeze() for xyz in xyzs.cpu().numpy()]

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
            clean_xyz(xyz=xyz, mask=mask_extract_boundary(mask)),
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
        max_distance=0.1,
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
    # test_hydra_centroid_alignment()
    # test_hydra_correspondence_indices()
    # test_hydra_icp()
    test_hydra_robust_icp()
