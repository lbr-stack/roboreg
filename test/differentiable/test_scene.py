import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import cv2
import numpy as np
import torch

from roboreg import differentiable as rrd
from roboreg.io import URDFParser, find_files, parse_camera_info
from roboreg.util import overlay_mask


def test_single_config_stereo_view() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_link_name = "link_0"
    end_link_name = "link_7"

    data_prefix = "test/data/lbr_med7"
    recording_prefix = "zed2i/zurich_calibration_data"
    prefix = os.path.join(data_prefix, recording_prefix)

    # initial transform
    camera_names = ["left", "right"]
    ht_base_left = np.load(os.path.join(prefix, "HT_hydra_robust.npy"))
    ht_right_left = np.load(
        os.path.join(
            prefix, "HT_zed_bench_right_camera_frame_to_zed_bench_left_camera_frame.npy"
        )
    )
    extrinsics = {
        camera_names[0]: ht_base_left,
        camera_names[1]: ht_base_left @ ht_right_left,
    }

    # instantiate cameras and load masks
    cameras = {}
    masks = {}
    images = {}
    for camera_name in camera_names:
        height, width, intrinsics = parse_camera_info(
            os.path.join(prefix, f"{camera_name}_camera_info.yaml")
        )
        cameras[camera_name] = rrd.VirtualCamera(
            intrinsics=intrinsics,
            extrinsics=extrinsics[camera_name],
            resolution=[height, width],
            device=device,
        )

        masks[camera_name] = [
            cv2.imread(os.path.join(prefix, file), cv2.IMREAD_GRAYSCALE)
            for file in find_files(prefix, f"{camera_name}_masked_*.png")
        ]

        images[camera_name] = [
            cv2.imread(os.path.join(prefix, file))
            for file in find_files(prefix, f"{camera_name}_img_*.png")
        ]

    # load joint states
    joint_states = [
        np.load(os.path.join(prefix, file))
        for file in find_files(prefix, "joint_state_*.npy")
    ]

    # to tensors
    for camera_name in camera_names:
        masks[camera_name] = (
            torch.tensor(
                np.array(masks[camera_name]), device=device, dtype=torch.float32
            )
            / 255.0
        )

    joint_states = torch.tensor(
        np.array(joint_states), device=device, dtype=torch.float32
    )

    # instantiate URDF parser
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )

    # instantiate meshes
    meshes = rrd.TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name
        ),
        batch_size=joint_states.shape[0],
        device=device,
    )

    # instantiate kinematics
    kinematics = rrd.TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )

    # instantiate renderer
    renderer = rrd.NVDiffRastRenderer(
        device=device,
    )

    # instantiate scene
    scene = rrd.RobotScene(
        meshes=meshes,
        kinematics=kinematics,
        renderer=renderer,
        cameras=cameras,
    )

    scene.configure_robot_joint_states(q=joint_states)

    for camera_name in camera_names:
        # scene.configure_camera(camera_name)   ### TODO: configure camera pose
        renders = scene.observe_from(camera_name)

        # show first render and overlay
        for render, image in zip(renders, images[camera_name]):
            render = render.squeeze().cpu().numpy()
            overlay = overlay_mask(
                image,
                (render * 255.0).astype(np.uint8),
                scale=1.0,
            )

            cv2.imshow(camera_name, overlay)
            cv2.waitKey(0)


def test_single_config_stereo_view_pose_optimization() -> None:
    pass


if __name__ == "__main__":
    test_single_config_stereo_view()
    # test_single_config_stereo_view_pose_optimization()
