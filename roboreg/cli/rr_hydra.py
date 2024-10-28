import argparse
import os

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


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="image_*_mask.png",
        help="Mask file pattern.",
    )
    parser.add_argument(
        "--xyz-pattern", type=str, default="xyz_*.npy", help="XYZ file pattern."
    )
    parser.add_argument(
        "--joint-states-pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--ros-package",
        type=str,
        default="lbr_description",
        help="Package where the URDF is located.",
    )
    parser.add_argument(
        "--xacro-path",
        type=str,
        default="urdf/med7/med7.xacro",
        help="Path to the xacro file, relative to --ros-package.",
    )
    parser.add_argument(
        "--root-link-name", type=str, default="lbr_link_0", help="Root link name."
    )
    parser.add_argument(
        "--end-link-name", type=str, default="lbr_link_7", help="End link name."
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=5000,
        help="Number of points to sample from robot mesh.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.01,
        help="Maximum distance between two points to be considered as a correspondence.",
    )
    parser.add_argument(
        "--outer-max-iter",
        type=int,
        default=50,
        help="Maximum number of outer iterations.",
    )
    parser.add_argument(
        "--inner-max-iter",
        type=int,
        default=10,
        help="Maximum number of inner iterations.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="HT_hydra_robust.npy",
        help="Output file name. Relative to the path.",
    )
    parser.add_argument(
        "--erosion-kernel-size",
        type=int,
        default=10,
        help="Erosion kernel size for mask boundary. Larger value will result in larger boundary. The closer the robot, the larger the recommended kernel size.",
    )
    return parser.parse_args()


def main():
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    joint_states, masks, xyzs = parse_hydra_data(
        path=args.path,
        joint_states_pattern=args.joint_states_pattern,
        mask_pattern=args.mask_pattern,
        xyz_pattern=args.xyz_pattern,
    )

    # instantiate kinematics
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(ros_package=args.ros_package, xacro_path=args.xacro_path)
    kinematics = TorchKinematics(
        urdf_parser=urdf_parser,
        device=device,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
    )

    # instantiate mesh
    batch_size = len(joint_states)
    meshes = TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths(
            root_link_name=args.root_link_name, end_link_name=args.end_link_name
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

    # sample N points per mesh
    for i in range(batch_size):
        idx = torch.randperm(mesh_vertices[i].shape[0])[: args.number_of_points]
        mesh_vertices[i] = mesh_vertices[i][idx]
        mesh_normals[i] = mesh_normals[i][idx]

    HT_init = hydra_centroid_alignment(observed_vertices, mesh_vertices)
    HT = hydra_robust_icp(
        HT_init,
        observed_vertices,
        mesh_vertices,
        mesh_normals,
        max_distance=args.max_distance,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
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
    np.save(os.path.join(args.path, args.output_file), HT)


if __name__ == "__main__":
    main()
