import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from typing import List

import cv2
import numpy as np
import torch
import transformations as tf
import trimesh

from roboreg.differentiable.renderer import NVDiffRastRenderer
from roboreg.differentiable.structures import TorchRobotMesh


def test_nvdiffrast_pose_optimization() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh_paths: List[trimesh.Geometry] = [
        trimesh.load(f"test/data/lbr_med7/mesh/link_{idx}.stl") for idx in range(8)
    ]
    torch_robot_mesh = TorchRobotMesh(mesh_paths=mesh_paths, device=device)
    renderer = NVDiffRastRenderer(device=device)

    # transform mesh to it becomes visible
    HT_TARGET = torch.tensor(
        tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32"),
        device=device,
        dtype=torch.float32,
    )
    torch_robot_mesh.vertices = torch.matmul(torch_robot_mesh.vertices, HT_TARGET.T)

    # create a target render
    resolution = [256, 256]
    target_render = renderer.constant_color(
        torch_robot_mesh.vertices, torch_robot_mesh.faces, resolution
    )

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
        for i in range(1000):
            vertices = torch.matmul(torch_robot_mesh.vertices, HT.T)
            current_render = renderer.constant_color(
                vertices, torch_robot_mesh.faces, resolution
            )
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


if __name__ == "__main__":
    test_nvdiffrast_pose_optimization()
