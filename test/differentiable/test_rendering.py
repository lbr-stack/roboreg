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
from roboreg.io import URDFParser, find_files, parse_camera_info
from roboreg.util import overlay_mask


def test_unit() -> None:
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
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate meshes
    meshes = TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths("link_0", "link_7"),
        device=device,
    )

    base_link_vertices = meshes.get_mesh_vertices(meshes.mesh_names[0]).clone()
    print(base_link_vertices.mean(dim=1))

    # load camera intrinsics
    height, width, intrinsics_3x3 = parse_camera_info(
        "test/data/lbr_med7/zed2i/high_res/camera_info.yaml"
    )
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = intrinsics_3x3
    intrinsics = torch.tensor(intrinsics, device=device, dtype=torch.float32)

    # instantiate renderer
    renderer = NVDiffRastRenderer(device=device)

    # project points
    data_prefix = "test/data/lbr_med7/zed2i/high_res"
    ht_base_cam = np.load(os.path.join(data_prefix, "HT_hydra_robust.npy"))

    # static transforms
    ht_cam_optical = tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # camera -> optical

    # base to optical frame
    ht_base_optical = ht_base_cam @ ht_cam_optical  # base frame -> optical
    ht_optical_base = np.linalg.inv(ht_base_optical)
    ht_optical_base = torch.tensor(ht_optical_base, device=device, dtype=torch.float32)

    # http://www.songho.ca/opengl/gl_projectionmatrix.html
    # http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    # https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ
    # https://sightations.wordpress.com/2010/08/03/simulating-calibrated-cameras-in-opengl/
    # 2*fx/W, 2*s/W , 2*(cx/W)-1             , 0                       | x
    # 0     , 2*fy/H, 2*(cy/H)-1             , 0                       | y
    # 0     , 0     , (zmax+zmin)/(zmax-zmin), 2*zmax*zmin/(zmin-zmax) | z
    # 0     , 0     , 1                      , 0                       | w
    zmin = 0.1
    zmax = 10.0

    projection = torch.tensor(
        [
            [
                2.0 * intrinsics[0, 0] / width,
                0.0,
                2.0 * intrinsics[0, 2] / width - 1.0,
                0.0,
            ],
            [
                0.0,
                2.0 * intrinsics[1, 1] / height,
                2.0 * intrinsics[1, 2] / height - 1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                (zmax + zmin) / (zmax - zmin),
                2.0 * zmax * zmin / (zmin - zmax),
            ],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    print("projection matrix")
    print(projection)
    meshes.vertices = torch.matmul(
        meshes.vertices, ht_optical_base.T @ projection.T
    )  # perform projection to clip space

    # render
    resolution = [height + 4, width]  # hack so divisible by 8
    render = renderer.constant_color(
        meshes.vertices,
        meshes.faces,
        resolution=resolution,
    )

    render = render.detach().cpu().numpy().squeeze()
    render = render[2:-2, :]  # crop the entire solution but crop for now

    image = cv2.imread(os.path.join(data_prefix, "image_0.png"))
    print(image.shape)
    overlay = overlay_mask(image, (render * 255).astype(np.uint8), scale=1.0)

    cv2.imshow("image", image)
    cv2.imshow("render", render)
    cv2.imshow("overlay", overlay)

    cv2.waitKey(0)


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
    # test_unit()
    test_single_view_rendering()
    # test_nvdiffrast_simple_pose_optimization()
    # test_multi_config_single_view_rendering()
    # test_nvdiffrast_multi_config_single_view_pose_optimization()
