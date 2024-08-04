import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from typing import List

import cv2
import numpy as np
import torch
import transformations as tf
import trimesh
from tqdm import tqdm

from roboreg.differentiable.kinematics import TorchKinematics
from roboreg.differentiable.rendering import NVDiffRastRenderer
from roboreg.differentiable.structs import TorchMeshContainer
from roboreg.io import URDFParser


def test_nvdiffrast_simple_pose_optimization() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh_paths: List[trimesh.Geometry] = {
        f"link_{idx}": f"test/data/lbr_med7/mesh/link_{idx}.stl" for idx in range(8)
    }
    meshes = TorchMeshContainer(mesh_paths=mesh_paths, device=device)
    renderer = NVDiffRastRenderer(device=device)

    # transform mesh to it becomes visible
    HT_TARGET = torch.tensor(
        tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32"),
        device=device,
        dtype=torch.float32,
    )
    meshes.vertices = torch.matmul(meshes.vertices, HT_TARGET.T)

    # create a target render
    resolution = [512, 512]
    target_render = renderer.constant_color(meshes.vertices, meshes.faces, resolution)

    # modify transform
    HT = torch.tensor(
        tf.euler_matrix(0.0, 0.0, np.pi / 16.0).astype("float32"),
        device=device,
        requires_grad=True,
        dtype=torch.float32,
    )

    # create an optimizer an optimize HT -> HT_TARGET
    optimizer = torch.optim.Adam([HT], lr=0.001)
    metric = torch.nn.MSELoss()
    try:
        for _ in tqdm(range(1000)):
            vertices = torch.matmul(meshes.vertices, HT.T)
            current_render = renderer.constant_color(vertices, meshes.faces, resolution)
            loss = metric(current_render, target_render)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # visualize
            current_image = current_render.squeeze().cpu().detach().numpy()
            target_image = target_render.squeeze().cpu().numpy()
            difference_image = current_image - target_image
            concatenated_image = np.concatenate(
                [target_image, current_image, difference_image], axis=-1
            )
            cv2.imshow(
                "target render / current render / difference", concatenated_image
            )
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass


def test_multi_config_rendering() -> None:
    urder_parser = URDFParser()
    urder_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate kinematics / meshes / renderer
    kinematics = TorchKinematics(
        urdf=urder_parser.urdf,
        root_link_name="link_0",
        end_link_name="link_7",
        device=device,
    )

    batch_size = 4

    meshes = TorchMeshContainer(
        mesh_paths=urder_parser.ros_package_mesh_paths("link_0", "link_7"),
        batch_size=batch_size,
        device=device,
    )

    renderer = NVDiffRastRenderer(device=device)

    # create batches joint states
    q = torch.zeros([batch_size, kinematics.chain.n_joints], device=device)
    q[0, 1] = 0.0
    q[1, 1] = 1.0 / 3.0 * torch.pi / 2.0
    q[2, 1] = 2.0 / 3.0 * torch.pi / 2.0
    q[3, 1] = 3.0 / 3.0 * torch.pi / 2.0

    # compute batched forward kinematics
    ht_lookup = kinematics.mesh_forward_kinematics(q)
    ht_view = torch.tensor(
        tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32"),
        device=device,
        dtype=torch.float32,
    )

    # apply batched forward kinematics
    for link_name, ht in ht_lookup.items():
        print(link_name)
        print(ht.shape)
        meshes.set_mesh_vertices(
            link_name,
            torch.matmul(
                meshes.get_mesh_vertices(link_name),
                ht.transpose(-1, -2),
            ),
        )

    meshes.vertices = torch.matmul(meshes.vertices, ht_view.T)

    # render batch
    renders = renderer.constant_color(
        meshes.vertices,
        meshes.faces,
        [256, 256],
    )

    print(renders.shape)

    for idx, render in enumerate(renders):
        cv2.imshow(f"render_{idx}", render.detach().cpu().numpy().squeeze())
    cv2.waitKey(0)


def test_nvdiffrast_stereo_multi_config_pose_optimization() -> None:
    ### take hydra ICP as initial guess

    ### implement stereo multi-config pose optimization

    ### render results and compare them to hydra ICP only

    ####################### Finally....
    ### use this method to refine masks
    ### attempt to fix synchronization issues
    ### train segmentation model (left / right)

    pass


if __name__ == "__main__":
    # test_nvdiffrast_simple_pose_optimization()
    test_multi_config_rendering()
    # test_nvdiffrast_stereo_multi_config_pose_optimization()
