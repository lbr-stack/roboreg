import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)


import cv2
import numpy as np
import torch
import transformations as tf
from tqdm import tqdm

from roboreg import differentiable as rrd
from roboreg.io import URDFParser, find_files, parse_camera_info
from roboreg.util import overlay_mask


class TestRendering:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        data_prefix: str = "test/data/lbr_med7",
        recording_prefix: str = "zed2i/high_res",
        root_link_name: str = "lbr_link_0",
        end_link_name: str = "lbr_link_7",
        batch_size: int = 1,
    ) -> None:
        # setup
        self.device = device
        self.data_prefix = data_prefix
        self.recording_prefix = recording_prefix
        self.root_link_name = root_link_name
        self.end_link_name = end_link_name
        self.batch_size = batch_size

        # load data
        prefix = os.path.join(self.data_prefix, self.recording_prefix)
        self.images = [
            cv2.imread(os.path.join(prefix, file))
            for file in find_files(prefix, "image_*.png")
        ]
        self.masks = [
            cv2.imread(os.path.join(prefix, file), cv2.IMREAD_GRAYSCALE)
            for file in find_files(prefix, "mask_*.png")
        ]
        self.joint_states = [
            np.load(os.path.join(prefix, file))
            for file in find_files(
                prefix,
                "joint_states_*.npy",
            )
        ]

        if len(self.images) != len(self.masks) or len(self.images) != len(
            self.joint_states
        ):
            raise RuntimeError(
                "Expected same number of images, masks, and joint states."
            )

        # instantiate URDF parser
        self.urder_parser = URDFParser()
        self.urder_parser.from_ros_xacro(
            ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
        )

        # instantiate meshes
        self.meshes = rrd.TorchMeshContainer(
            mesh_paths=self.urder_parser.ros_package_mesh_paths(
                self.root_link_name, self.end_link_name
            ),
            batch_size=self.batch_size,
            device=self.device,
        )

        # instantiante kinematics
        self.kinematics = rrd.TorchKinematics(
            urdf=self.urder_parser.urdf,
            root_link_name=self.root_link_name,
            end_link_name=self.end_link_name,
            device=self.device,
        )

        # load camera intrinsics and initial extrinsics
        self.height, self.width, self.intrinsics = parse_camera_info(
            os.path.join(prefix, "camera_info.yaml")
        )

        # instantiate camera and renderer
        self.ht_base_cam = np.load(os.path.join(prefix, "HT_hydra_robust.npy"))
        self.renderer = rrd.NVDiffRastRenderer(device=self.device)


def test_nvdiffrast_unit() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vertices = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 1.0],
                [-0.5, 0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, 1.0],
            ],  # x-y plane
        ],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)

    renderer = rrd.NVDiffRastRenderer()
    render = renderer.constant_color(vertices, faces, [256, 256])

    cv2.imshow("render", render.cpu().numpy().squeeze())
    cv2.waitKey(0)


def test_single_view_rendering() -> None:
    test_rendering = TestRendering()
    data_idx = 2

    # compute and apply kinematics
    ht_lookup = test_rendering.kinematics.mesh_forward_kinematics(
        torch.tensor(
            test_rendering.joint_states[data_idx],
            dtype=torch.float32,
            device=test_rendering.device,
        )
    )

    # create copy of vertices and keep container intact
    cloned_vertices = test_rendering.meshes.vertices.clone()

    # apply homogeneous transforms
    for link_name, ht in ht_lookup.items():
        cloned_vertices[
            :,
            test_rendering.meshes.lower_vertex_index_lookup[
                link_name
            ] : test_rendering.meshes.upper_vertex_index_lookup[link_name],
        ] = torch.matmul(
            cloned_vertices[
                :,
                test_rendering.meshes.lower_vertex_index_lookup[
                    link_name
                ] : test_rendering.meshes.upper_vertex_index_lookup[link_name],
            ],
            ht.transpose(-1, -2),
        )

    # create a virtual camera
    camera = rrd.VirtualCamera(
        resolution=[test_rendering.height, test_rendering.width],
        intrinsics=test_rendering.intrinsics,
        extrinsics=test_rendering.ht_base_cam,
        device=test_rendering.device,
    )

    # project points
    cloned_vertices = torch.matmul(
        cloned_vertices,
        torch.linalg.inv(camera.extrinsics @ camera.ht_optical).T
        @ camera.perspective_projection.T,
    )  # perform projection to clip space

    # render
    render = test_rendering.renderer.constant_color(
        clip_vertices=cloned_vertices,
        faces=test_rendering.meshes.faces,
        resolution=camera.resolution,
    )

    render = render.detach().cpu().numpy().squeeze()

    overlay = overlay_mask(
        test_rendering.images[data_idx], (render * 255).astype(np.uint8), scale=1.0
    )

    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)


def test_single_config_single_view_pose_optimization() -> None:
    test_rendering = TestRendering()
    data_idx = 2

    # compute and apply kinematics
    ht_lookup = test_rendering.kinematics.mesh_forward_kinematics(
        torch.tensor(
            test_rendering.joint_states[data_idx],
            dtype=torch.float32,
            device=test_rendering.device,
        )
    )

    for link_name, ht in ht_lookup.items():
        test_rendering.meshes.transform_mesh(ht, link_name)

    # create differentiable camera and initialize extrinsics
    camera = rrd.VirtualCamera(
        resolution=[test_rendering.height, test_rendering.width],
        intrinsics=test_rendering.intrinsics,
        extrinsics=torch.tensor(
            test_rendering.ht_base_cam,
            dtype=torch.float32,
            device=test_rendering.device,
            requires_grad=True,
        ),
        device=test_rendering.device,
    )

    # create an optimizer and optimize intrinsics
    optimizer = torch.optim.SGD([camera.extrinsics], lr=0.001)
    metric = torch.nn.BCELoss()

    # load target
    target_mask = (
        torch.tensor(
            test_rendering.masks[data_idx],
            dtype=torch.float32,
            device=test_rendering.device,
        )
        .unsqueeze(0)
        .unsqueeze(-1)
        / 255.0
    )

    try:
        for _ in tqdm(range(1000)):
            vertices = torch.matmul(
                test_rendering.meshes.vertices,
                torch.linalg.inv(camera.extrinsics @ camera.ht_optical).T
                @ camera.perspective_projection.T,
            )
            current_mask = test_rendering.renderer.constant_color(
                vertices, test_rendering.meshes.faces, camera.resolution
            )
            loss = metric(current_mask, target_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # visualize
            current_mask = current_mask.squeeze().cpu().detach().numpy()
            overlay = overlay_mask(
                test_rendering.images[data_idx],
                (current_mask * 255.0).astype(np.uint8),
                scale=1.0,
            )
            cv2.imshow("overlay", overlay)
            cv2.waitKey(0)
    except KeyboardInterrupt:
        pass


def test_multi_config_single_view_rendering() -> None:
    test_rendering = TestRendering()

    # create batched joint states
    q = torch.tensor(
        np.array(test_rendering.joint_states),
        dtype=torch.float32,
        device=test_rendering.device,
    )

    # overwrite meshes with batch size
    test_rendering.meshes = rrd.TorchMeshContainer(
        mesh_paths=test_rendering.urder_parser.ros_package_mesh_paths(
            test_rendering.root_link_name, test_rendering.end_link_name
        ),
        batch_size=q.shape[0],
        device=test_rendering.device,
    )

    # compute batched forward kinematics
    ht_lookup = test_rendering.kinematics.mesh_forward_kinematics(q)

    # apply batched forward kinematics
    for link_name, ht in ht_lookup.items():
        test_rendering.meshes.transform_mesh(ht, link_name)

    # create a virtual camera
    camera = rrd.VirtualCamera(
        resolution=[test_rendering.height, test_rendering.width],
        intrinsics=test_rendering.intrinsics,
        extrinsics=test_rendering.ht_base_cam,
        device=test_rendering.device,
    )

    # apply common view to all
    test_rendering.meshes.vertices = torch.matmul(
        test_rendering.meshes.vertices,
        torch.linalg.inv(camera.extrinsics @ camera.ht_optical).T
        @ camera.perspective_projection.T,
    )

    # render batch
    renders = test_rendering.renderer.constant_color(
        test_rendering.meshes.vertices,
        test_rendering.meshes.faces,
        resolution=camera.resolution,
    )

    for idx, render in enumerate(renders):
        cv2.imshow(f"render_{idx}", render.detach().cpu().numpy().squeeze())
    cv2.waitKey(0)


def test_multi_config_single_view_pose_optimization() -> None:
    test_rendering = TestRendering()

    # convert to tensors
    target_masks = (
        torch.tensor(
            np.array(test_rendering.masks),
            dtype=torch.float32,
            device=test_rendering.device,
        )
        / 255.0
    ).unsqueeze(-1)
    q = torch.tensor(
        np.array(test_rendering.joint_states),
        dtype=torch.float32,
        device=test_rendering.device,
    )

    # overwrite meshes with batch size
    test_rendering.meshes = rrd.TorchMeshContainer(
        mesh_paths=test_rendering.urder_parser.ros_package_mesh_paths(
            test_rendering.root_link_name, test_rendering.end_link_name
        ),
        batch_size=q.shape[0],
        device=test_rendering.device,
    )

    # compute and apply kinematics
    ht_lookup = test_rendering.kinematics.mesh_forward_kinematics(q)

    for link_name, ht in ht_lookup.items():
        test_rendering.meshes.transform_mesh(ht, link_name)

    # create differentiable camera and initialize extrinsics
    camera = rrd.VirtualCamera(
        resolution=[test_rendering.height, test_rendering.width],
        intrinsics=test_rendering.intrinsics,
        extrinsics=torch.tensor(
            test_rendering.ht_base_cam,
            dtype=torch.float32,
            device=test_rendering.device,
            requires_grad=True,
        ),
        device=test_rendering.device,
    )

    # create an optimizer and optimize intrinsics
    optimizer = torch.optim.SGD([camera.extrinsics], lr=0.001)
    metric = torch.nn.BCELoss()

    from torchvision.utils import (
        make_grid,
    )  # TODO: this is not a dependency except for this test

    try:
        for _ in tqdm(range(200)):
            vertices = torch.matmul(
                test_rendering.meshes.vertices,
                torch.linalg.inv(camera.extrinsics @ camera.ht_optical).T
                @ camera.perspective_projection.T,
            )
            current_masks = test_rendering.renderer.constant_color(
                vertices, test_rendering.meshes.faces, camera.resolution
            )
            loss = metric(current_masks, target_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # visualize
        current_masks = current_masks.squeeze().cpu().detach().numpy()
        overlays = []
        for current_mask, image in zip(current_masks, test_rendering.images):
            overlay = overlay_mask(
                image,
                (current_mask * 255.0).astype(np.uint8),
                scale=1.0,
            )
            overlays.append(cv2.resize(overlay, [256, 256]))

        overlays = torch.tensor(np.array(overlays)).permute(0, 3, 1, 2)
        overlays = make_grid(overlays, nrow=3)

        cv2.imshow("overlay", overlays.permute(1, 2, 0).numpy())
        cv2.waitKey(0)
    except KeyboardInterrupt:
        pass


def test_multi_camera_pose_rendering() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2  # render 2 cameras at once

    # load a sample mesh
    meshes = rrd.TorchMeshContainer(
        {"link_0": "test/data/lbr_med7/mesh/link_0.stl"},
        batch_size=batch_size,
        device=device,
    )

    # create a virtual camera
    resolution = (256, 256)
    intrinsics = torch.tensor(
        [
            [resolution[0], 0, resolution[0] / 2 - 1],
            [0, resolution[1], resolution[1] / 2 - 1],
            [0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )  # on purpose: intrinsics 1x3x3, extrinsics Bx4x4
    extrinsics = torch.zeros(batch_size, 4, 4, device=device, dtype=torch.float32)

    # create 2 unique views
    extrinsics[0] = torch.tensor(
        tf.euler_matrix(0.0, 0.0, 0.0).astype("float32"),
    )
    extrinsics[0, 2, 3] = -1
    extrinsics[1] = torch.tensor(
        tf.euler_matrix(0.0, 0.0, -np.pi / 4).astype("float32"),
    )
    extrinsics[1, 2, 3] = -1

    print(extrinsics[0])

    batched_camera = rrd.VirtualCamera(
        resolution=resolution, intrinsics=intrinsics, extrinsics=extrinsics
    )

    # renderer
    renderer = rrd.NVDiffRastRenderer(device=device)

    # project meshes (knowing that intrinsics are identity)
    vertices = meshes.vertices.clone()
    vertices = torch.matmul(
        torch.matmul(
            vertices,
            torch.linalg.inv(
                torch.matmul(
                    batched_camera.extrinsics.transpose(-1, -2),
                    batched_camera.ht_optical.transpose(-1, -2),
                ),
            ),
        ),
        batched_camera.perspective_projection.transpose(-1, -2),
    )

    # render
    renders = renderer.constant_color(
        clip_vertices=vertices,
        faces=meshes.faces,
        resolution=resolution,
    )

    for idx, render in enumerate(renders):
        cv2.imshow(f"render_{idx}", render.detach().cpu().numpy().squeeze())
    cv2.waitKey(0)


if __name__ == "__main__":
    # test_nvdiffrast_unit()
    # test_single_view_rendering()
    # test_single_config_single_view_pose_optimization()
    # test_multi_config_single_view_rendering()
    # test_multi_config_single_view_pose_optimization()
    test_multi_camera_pose_rendering()
