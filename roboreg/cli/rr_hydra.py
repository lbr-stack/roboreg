import argparse
import os

import numpy as np
import torch

from roboreg.hydra_icp import hydra_centroid_alignment, hydra_robust_icp
from roboreg.io import load_data, visualize_registration
from roboreg.util import find_files


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--mask_pattern",
        type=str,
        default="mask_*.png",
        help="Mask file pattern.",
    )
    parser.add_argument(
        "--xyz_pattern", type=str, default="xyz_*.npy", help="XYZ file pattern."
    )
    parser.add_argument(
        "--joint_states_pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--number_of_points",
        type=int,
        default=5000,
        help="Number of points to sample from robot mesh.",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=0.01,
        help="Maximum distance between two points to be considered as a correspondence.",
    )
    parser.add_argument(
        "--outer_max_iter",
        type=int,
        default=50,
        help="Maximum number of outer iterations.",
    )
    parser.add_argument(
        "--inner_max_iter",
        type=int,
        default=10,
        help="Maximum number of inner iterations.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="HT_hydra_robust.npy",
        help="Output file name. Relative to the path.",
    )
    parser.add_argument(
        "--convex_hull",
        action="store_true",
        help="Use convex hull for collision mesh.",
    )
    parser.add_argument(
        "--erosion_kernel_size",
        type=int,
        default=10,
        help="Erosion kernel size for mask boundary.",
    )
    return parser.parse_args()


def main():
    args = args_factory()

    path = args.path
    mask_files = find_files(path, args.mask_pattern)
    xyz_files = find_files(path, args.xyz_pattern)
    joint_state_files = find_files(path, args.joint_states_pattern)
    number_of_points = args.number_of_points

    observed_xyzs, mesh_xyzs, mesh_xyzs_normals = load_data(
        path=path,
        mask_files=mask_files,
        xyz_files=xyz_files,
        joint_state_files=joint_state_files,
        number_of_points=number_of_points,
        erosion_kernel_size=args.erosion_kernel_size,
        convex_hull=args.convex_hull,
    )
    if len(observed_xyzs) == 0 or len(mesh_xyzs) == 0 or len(mesh_xyzs_normals) == 0:
        raise ValueError("Failed to load data.")

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
        max_distance=args.max_distance,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
    )

    # to numpy
    HT = HT.cpu().numpy()
    np.save(os.path.join(path, args.output_file), HT)

    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = observed_xyzs[i].cpu().numpy()
        mesh_xyzs[i] = mesh_xyzs[i].cpu().numpy()

    visualize_registration(observed_xyzs, mesh_xyzs, np.linalg.inv(HT))


if __name__ == "__main__":
    main()
