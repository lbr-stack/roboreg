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
from roboreg.differentiable.structs import TorchMeshContainer, VirtualCamera
from roboreg.io import URDFParser, find_files, parse_camera_info
from roboreg.util import overlay_mask


class TestRenderingFixture:
    def __init__(self) -> None:
        # setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_prefix = "test/data/lbr_med7"
        self.recording_prefix = "zed2i/high_res"
        self.root_link_name = "link_0"
        self.end_link_name = "link_7"

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

        # instantiate URDF parser
        self.urder_parser = URDFParser()
        self.urder_parser.from_ros_xacro(
            ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
        )

        # instantiate meshes
        self.meshes = TorchMeshContainer(
            mesh_paths=self.urder_parser.ros_package_mesh_paths(
                self.root_link_name, self.end_link_name
            ),
            device=self.device,
        )

        # instantiante kinematics
        self.kinematics = TorchKinematics(
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
        self.renderer = NVDiffRastRenderer(device=self.device)


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
        device=device,
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)

    renderer = NVDiffRastRenderer()
    render = renderer.constant_color(vertices, faces, [256, 256])

    cv2.imshow("render", render.cpu().numpy().squeeze())
    cv2.waitKey(0)


def test_single_view_rendering() -> None:
    test_fixture = TestRenderingFixture()
    data_idx = 2

    # compute and apply kinematics
    ht_lookup = test_fixture.kinematics.mesh_forward_kinematics(
        torch.tensor(
            test_fixture.joint_states[data_idx],
            device=test_fixture.device,
            dtype=torch.float32,
        )
    )

    for link_name, ht in ht_lookup.items():
        test_fixture.meshes.set_mesh_vertices(
            link_name,
            torch.matmul(
                test_fixture.meshes.get_mesh_vertices(link_name),
                ht.transpose(-1, -2),
            ),
        )

    camera = VirtualCamera(
        intrinsics=test_fixture.intrinsics,
        extrinsics=test_fixture.ht_base_cam,
        resolution=[test_fixture.height, test_fixture.width],
        device=test_fixture.device,
    )

    # project points
    test_fixture.meshes.vertices = torch.matmul(
        test_fixture.meshes.vertices,
        torch.linalg.inv(camera.extrinsics @ camera.ht_optical).T
        @ camera.perspective_projection.T,
    )  # perform projection to clip space

    # render
    render = test_fixture.renderer.constant_color(
        clip_vertices=test_fixture.meshes.vertices,
        faces=test_fixture.meshes.faces,
        resolution=[camera.width, camera.height],
    )

    render = render.detach().cpu().numpy().squeeze()

    overlay = overlay_mask(
        test_fixture.images[data_idx], (render * 255).astype(np.uint8), scale=1.0
    )

    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)


def test_single_config_single_view_pose_optimization() -> None:
    test_fixture = TestRenderingFixture()
    data_idx = 2

    # compute and apply kinematics
    ht_lookup = test_fixture.kinematics.mesh_forward_kinematics(
        torch.tensor(
            test_fixture.joint_states[data_idx],
            device=test_fixture.device,
            dtype=torch.float32,
        )
    )

    for link_name, ht in ht_lookup.items():
        test_fixture.meshes.set_mesh_vertices(
            link_name,
            torch.matmul(
                test_fixture.meshes.get_mesh_vertices(link_name),
                ht.transpose(-1, -2),
            ),
        )

    # create differentiable camera and initialize extrinsics
    camera = VirtualCamera(
        intrinsics=test_fixture.intrinsics,
        extrinsics=torch.tensor(
            test_fixture.ht_base_cam,
            device=test_fixture.device,
            dtype=torch.float32,
            requires_grad=True,
        ),
        resolution=[test_fixture.height, test_fixture.width],
        device=test_fixture.device,
    )

    # create an optimizer and optimize intrinsics
    optimizer = torch.optim.Adam([camera.extrinsics], lr=0.001)
    metric = torch.nn.BCELoss()

    # load target
    target_mask = (
        torch.tensor(
            test_fixture.masks[data_idx],
            device=test_fixture.device,
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .unsqueeze(-1)
        / 255.0
    )

    try:
        for _ in tqdm(range(1000)):
            vertices = torch.matmul(
                test_fixture.meshes.vertices,
                torch.linalg.inv(camera.extrinsics @ camera.ht_optical).T
                @ camera.perspective_projection.T,
            )
            current_mask = test_fixture.renderer.constant_color(
                vertices, test_fixture.meshes.faces, camera.resolution
            )
            loss = metric(current_mask, target_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # visualize
            current_mask = current_mask.squeeze().cpu().detach().numpy()
            overlay = overlay_mask(
                test_fixture.images[data_idx],
                (current_mask * 255.0).astype(np.uint8),
                scale=1.0,
            )
            cv2.imshow("overlay", overlay)
            cv2.waitKey(0)
    except KeyboardInterrupt:
        pass


def test_multi_config_single_view_rendering() -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate kinematics / meshes / renderer
    kinematics = TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name="link_0",
        end_link_name="link_7",
        device=device,
    )

    n_configurations = 4

    meshes = TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths("link_0", "link_7"),
        batch_size=n_configurations,
        device=device,
    )

    renderer = NVDiffRastRenderer(device=device)

    # create batches joint states
    q = torch.zeros([n_configurations, kinematics.chain.n_joints], device=device)
    q[:, 1] = torch.linspace(0, torch.pi / 2.0, n_configurations)

    # compute batched forward kinematics
    ht_lookup = kinematics.mesh_forward_kinematics(q)
    ht_view = torch.tensor(
        tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32"),
        device=device,
        dtype=torch.float32,
    )

    # apply batched forward kinematics
    for link_name, ht in ht_lookup.items():
        meshes.set_mesh_vertices(
            link_name,
            torch.matmul(
                meshes.get_mesh_vertices(link_name),
                ht.transpose(-1, -2),
            ),
        )

    # apply common view to all
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


def test_nvdiffrast_multi_config_single_view_pose_optimization() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    #   - initial pose (e.g. as obtained through Hydra-ICP)
    #   - rgb images
    #   - masks
    #   - joint states
    #   - camera intrinsics
    data_prefix = "test/data/lbr_med7/zed2i/high_res"
    ht_view_init = np.load(os.path.join(data_prefix, "HT_hydra_robust.npy"))
    ht_view_init = torch.tensor(ht_view_init, device=device, dtype=torch.float32)

    images = [
        cv2.imread(os.path.join(data_prefix, file))
        for file in find_files(data_prefix, "image_*.png")
    ]
    masks = [
        cv2.imread(os.path.join(data_prefix, file), cv2.IMREAD_GRAYSCALE)
        for file in find_files(data_prefix, "mask_*.png")
    ]
    joint_states = [
        np.load(os.path.join(data_prefix, file))
        for file in find_files(data_prefix, "joint_states_*.npy")
    ]
    height, width, intrinsics_3x3 = parse_camera_info(
        os.path.join(data_prefix, "camera_info.yaml")
    )
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = intrinsics_3x3

    if len(images) != len(masks) or len(images) != len(joint_states):
        raise RuntimeError("Expected same number of images, masks, and joint states.")

    # convert to tensors
    images = torch.tensor(images, device=device, dtype=torch.float32)
    masks = torch.tensor(masks, device=device, dtype=torch.float32)
    joint_states = torch.tensor(joint_states, device=device, dtype=torch.float32)
    intrinsics = torch.tensor(intrinsics, device=device, dtype=torch.float32)

    # initialize renderer
    #   - load URDF
    #   - instantiate kinematics
    #   - instantiate meshes
    #   - instantiate renderer
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    kinematics = TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name="link_0",
        end_link_name="link_7",
        device=device,
    )
    n_configurations = joint_states.shape[0]
    meshes = TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths("link_0", "link_7"),
        batch_size=n_configurations,
        device=device,
    )
    renderer = NVDiffRastRenderer(device=device)

    # compute kinematics
    ht_lookup = kinematics.mesh_forward_kinematics(joint_states)
    for link_name, ht in ht_lookup.items():
        meshes.set_mesh_vertices(
            link_name,
            torch.matmul(
                meshes.get_mesh_vertices(link_name),
                ht.transpose(-1, -2),
            ),
        )

    # apply common view to all
    meshes.vertices = torch.matmul(meshes.vertices, torch.linalg.inv(ht_view_init).T)
    meshes.vertices = torch.matmul(meshes.vertices, intrinsics.T)
    meshes.vertices = meshes.vertices / meshes.vertices[:, -2, None]

    # normalize by width / height to [-1, 1]
    meshes.vertices[:, :, 0] = 2.0 * meshes.vertices[:, :, 0] / width - 1.0
    meshes.vertices[:, :, 1] = 2.0 * meshes.vertices[:, :, 1] / height - 1.0

    print(meshes.vertices)

    # meshes.vertices

    # normalize
    # print(meshes.vertices)
    # meshes.vertices = meshes.vertices / meshes.vertices[:, -2, None]
    print(meshes.vertices)
    # print(meshes.shape)

    # render batch
    # resolution = masks.shape[-2:] # must be divisible by 8!!! https://github.com/NVlabs/nvdiffrast/issues/193#issuecomment-2250239862
    resolution = [
        height + 4,  # hack so divisible by 8
        width,
    ]  # TODO: adjust points / intrinsics etc accordingly!
    renders = renderer.constant_color(
        meshes.vertices,
        meshes.faces,
        resolution=resolution,
    )
    renders = renders[:, 2:-2, :, :]  # note the entire solution but crop for now

    print(renders.shape)

    for idx, render in enumerate(renders):
        cv2.imshow(f"render_{idx}", render.detach().cpu().numpy().squeeze())
    cv2.waitKey(0)

    # implement stereo multi-config pose optimization

    # render results and compare them to hydra ICP only

    ####################### Finally....
    ### use this method to refine masks
    ### attempt to fix synchronization issues
    ### train segmentation model (left / right)

    pass


if __name__ == "__main__":
    # test_nvdiffrast_unit()
    # test_single_view_rendering()
    test_single_config_single_view_pose_optimization()
    # test_multi_config_single_view_rendering()
    # test_nvdiffrast_multi_config_single_view_pose_optimization()
