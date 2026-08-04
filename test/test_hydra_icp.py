from pathlib import Path

import numpy as np
import pytest
import torch
import transformations as tf

from roboreg.core import TorchKinematics, TorchMeshContainer
from roboreg.io import (
    URDFParser,
    apply_mesh_origins,
    find_files,
    load_meshes,
    parse_camera_info,
    parse_hydra_observations,
)
from roboreg.registration.point_cloud.hydra import (
    centroid_alignment,
    correspondence_indices,
    point_to_point_icp,
    point_to_plane_robust_icp,
)
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


@pytest.mark.skip(reason="To be fixed.")
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

    HT = centroid_alignment(mesh_centroids, observed_centroids)

    assert torch.allclose(HT, HT_random)


@pytest.mark.skip(reason="To be fixed.")
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
    observed_vertices = torch.rand(M, dim)
    reference_vertices = torch.rand(N, dim)  # e.g. the mesh vertices
    matchindices, mask = correspondence_indices(
        observed_vertices,
        reference_vertices,
        max_correspondence_distance=np.sqrt(dim) / 2.0,  # remove some elements randomly
    )
    test_index_shape(matchindices, mask, torch.Size([M]), N)

    # batched input
    batch_size = 2
    observed_vertices = torch.rand(batch_size, M, dim)
    reference_vertices = torch.rand(batch_size, N, dim)
    matchindices, mask = correspondence_indices(
        observed_vertices,
        reference_vertices,
        max_correspondence_distance=np.sqrt(dim) / 2.0,
    )
    test_index_shape(matchindices, mask, torch.Size([batch_size, M]), N)

    # test for inverted case
    M = 10
    N = 100

    observed_vertices = torch.rand(M, dim)
    reference_vertices = torch.rand(N, dim)
    matchindices, mask = correspondence_indices(
        observed_vertices,
        reference_vertices,
        max_correspondence_distance=np.sqrt(dim) / 2.0,
    )
    test_index_shape(matchindices, mask, torch.Size([M]), N)


@pytest.mark.skip(reason="To be fixed.")
def test_hydra_point_to_point_icp():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ros_package = "lbr_description"
    xacro_path = "urdf/med7/med7.xacro"
    root_link_name = "lbr_link_0"
    end_link_name = "lbr_link_7"
    path = Path("test/assets/lbr_med7_r800/samples")
    camera_info_file = "left_camera_info.yaml"
    joint_states_pattern = "joint_states_*.npy"
    mask_pattern = "mask_sam2_left_*.png"
    depth_pattern = "depth_*.npy"

    # load data
    observations = parse_hydra_observations(
        joint_states_files=find_files(path, joint_states_pattern),
        mask_files=find_files(path, mask_pattern),
        depth_files=find_files(path, depth_pattern),
    )
    height, width, intrinsics = parse_camera_info(
        camera_info_file=path / camera_info_file
    )

    # instantiate kinematics
    urdf_parser = URDFParser.from_ros_xacro(
        ros_package=ros_package, xacro_path=xacro_path
    )
    kinematics = TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )

    # instantiate mesh
    batch_size = len(observations.joint_states)
    meshes = TorchMeshContainer(
        meshes=apply_mesh_origins(
            meshes=load_meshes(
                urdf_parser.mesh_paths_from_ros_registry(
                    root_link_name=root_link_name, end_link_name=end_link_name
                )
            ),
            origins=urdf_parser.mesh_origins(
                root_link_name=root_link_name, end_link_name=end_link_name
            ),
        ),
        batch_size=batch_size,
        device=device,
    )

    # perform forward kinematics
    reference_vertices = meshes.vertices.clone()
    joint_states = torch.tensor(
        np.array(observations.joint_states), dtype=torch.float32, device=device
    )
    ht_lookup = kinematics.forward_kinematics(joint_states)
    for link_name, ht in ht_lookup.items():
        reference_vertices[
            :,
            meshes.lower_vertex_index_lookup[
                link_name
            ] : meshes.upper_vertex_index_lookup[link_name],
        ] = torch.matmul(
            reference_vertices[
                :,
                meshes.lower_vertex_index_lookup[
                    link_name
                ] : meshes.upper_vertex_index_lookup[link_name],
            ],
            ht.transpose(-1, -2),
        )
    reference_vertices = from_homogeneous(reference_vertices)

    # turn depths into xyzs
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    depths = torch.tensor(
        np.array(observations.depths), dtype=torch.float32, device=device
    )
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

    # reference vertices to list
    reference_vertices = [reference_vertices[i].contiguous() for i in range(batch_size)]

    # clean observed vertices and turn into tensor
    observed_vertices = [
        torch.tensor(
            clean_xyz(xyz=xyz, mask=mask_extract_boundary(mask)),
            dtype=torch.float32,
            device=device,
        )
        for xyz, mask in zip(xyzs, observations.masks)
    ]

    # sample 5000 points per mesh
    for i in range(batch_size):
        idx = torch.randperm(reference_vertices[i].shape[0])[:5000]
        reference_vertices[i] = reference_vertices[i][idx]

    HT_init = centroid_alignment(observed_vertices, reference_vertices)
    registration_result = point_to_point_icp(
        HT_init,
        observed_vertices,
        reference_vertices,
        max_correspondence_distance=0.1,
        max_iterations=int(1e3),
        rmse_change_tolerance=1e-8,
    )

    # visualize
    visualizer = RegistrationVisualizer()
    visualizer(mesh_vertices=reference_vertices, observed_vertices=observed_vertices)
    visualizer(
        mesh_vertices=reference_vertices,
        observed_vertices=observed_vertices,
        HT=torch.linalg.inv(registration_result.extrinsics),
    )

    # to numpy
    np.save(
        os.path.join(path, "HT_hydra.npy"), registration_result.extrinsics.cpu().numpy()
    )


@pytest.mark.skip(reason="To be fixed.")
def test_hydra_point_to_plane_robust_icp() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ros_package = "lbr_description"
    xacro_path = "urdf/med7/med7.xacro"
    root_link_name = "lbr_link_0"
    end_link_name = "lbr_link_7"
    path = Path("test/assets/lbr_med7_r800/samples")
    camera_info_file = "left_camera_info.yaml"
    joint_states_pattern = "joint_states_*.npy"
    mask_pattern = "mask_sam2_left_*.png"
    depth_pattern = "depth_*.npy"

    # load data
    observations = parse_hydra_observations(
        joint_states_files=find_files(path, joint_states_pattern),
        mask_files=find_files(path, mask_pattern),
        depth_files=find_files(path, depth_pattern),
    )
    height, width, intrinsics = parse_camera_info(
        camera_info_file=path / camera_info_file
    )

    # instantiate kinematics
    urdf_parser = URDFParser.from_ros_xacro(
        ros_package=ros_package, xacro_path=xacro_path
    )
    kinematics = TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )

    # instantiate mesh
    batch_size = len(observations.joint_states)
    meshes = TorchMeshContainer(
        meshes=apply_mesh_origins(
            meshes=load_meshes(
                urdf_parser.mesh_paths_from_ros_registry(
                    root_link_name=root_link_name, end_link_name=end_link_name
                )
            ),
            origins=urdf_parser.mesh_origins(
                root_link_name=root_link_name, end_link_name=end_link_name
            ),
        ),
        batch_size=batch_size,
        device=device,
    )

    # perform forward kinematics
    reference_vertices = meshes.vertices.clone()
    joint_states = torch.tensor(
        np.array(observations.joint_states), dtype=torch.float32, device=device
    )
    ht_lookup = kinematics.forward_kinematics(joint_states)
    for link_name, ht in ht_lookup.items():
        reference_vertices[
            :,
            meshes.lower_vertex_index_lookup[
                link_name
            ] : meshes.upper_vertex_index_lookup[link_name],
        ] = torch.matmul(
            reference_vertices[
                :,
                meshes.lower_vertex_index_lookup[
                    link_name
                ] : meshes.upper_vertex_index_lookup[link_name],
            ],
            ht.transpose(-1, -2),
        )

    # turn depths into xyzs
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    depths = torch.tensor(
        np.array(observations.depths), dtype=torch.float32, device=device
    )
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
    reference_vertices = from_homogeneous(reference_vertices)
    reference_vertices = [reference_vertices[i].contiguous() for i in range(batch_size)]
    reference_normals = []
    for i in range(batch_size):
        reference_normals.append(
            compute_vertex_normals(vertices=reference_vertices[i], faces=meshes.faces)
        )

    # clean observed vertices and turn into tensor
    observed_vertices = [
        torch.tensor(
            clean_xyz(xyz=xyz, mask=mask_extract_boundary(mask)),
            dtype=torch.float32,
            device=device,
        )
        for xyz, mask in zip(xyzs, observations.masks)
    ]

    # sample 5000 points per mesh
    for i in range(batch_size):
        idx = torch.randperm(reference_vertices[i].shape[0])[:5000]
        reference_vertices[i] = reference_vertices[i][idx]
        reference_normals[i] = reference_normals[i][idx]

    HT_init = centroid_alignment(observed_vertices, reference_vertices)
    registration_result = point_to_plane_robust_icp(
        HT_init,
        observed_vertices,
        reference_vertices,
        reference_normals,
        max_correspondence_distance=0.1,
        max_outer_iterations=50,
        max_inner_iterations=10,
    )

    # visualize
    visualizer = RegistrationVisualizer()
    visualizer(mesh_vertices=reference_vertices, observed_vertices=observed_vertices)
    visualizer(
        mesh_vertices=reference_vertices,
        observed_vertices=observed_vertices,
        HT=torch.linalg.inv(registration_result.extrinsics),
    )

    # to numpy
    np.save(
        os.path.join(path, "HT_hydra_robust.npy"),
        registration_result.extrinsics.cpu().numpy(),
    )


if __name__ == "__main__":
    import os
    import sys

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # test_hydra_centroid_alignment()
    # test_hydra_correspondence_indices()
    # test_hydra_point_to_point_icp()
    test_hydra_point_to_plane_robust_icp()
