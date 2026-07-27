from pathlib import Path

import numpy as np
import pytest

from roboreg.io import (
    URDFParser,
    find_files,
    parse_camera_info,
    parse_hydra_observations,
    parse_monocular_observations,
    parse_stereo_observations,
)


def test_urdf_parser_from_urdf_file() -> None:
    urdf_path = Path("test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf")
    urdf_parser = URDFParser.from_urdf_file(path=urdf_path)
    root_link_name = "lbr_link_0"
    end_link_name = "lbr_link_ee"
    chain_link_names = urdf_parser.chain_link_names(
        root_link_name=root_link_name, end_link_name=end_link_name
    )
    assert (
        chain_link_names[0] == root_link_name
    ), f"Expected {root_link_name} root link, got {chain_link_names[0]}."
    assert (
        chain_link_names[-1] == end_link_name
    ), f"Expected {end_link_name} end link, got {chain_link_names[-1]}."

    # try resolve paths to meshes
    mesh_uris = urdf_parser.mesh_uris(
        root_link_name=root_link_name, end_link_name=end_link_name
    )
    mesh_paths = URDFParser.resolve_relative_uris(
        uris=mesh_uris, base_path=urdf_path.parent
    )
    assert all(
        [mesh_paths[link].exists() for link in mesh_paths]
    ), "Expected all mesh paths to exist."

    # check origins
    mesh_origins = urdf_parser.mesh_origins(
        root_link_name=root_link_name, end_link_name=end_link_name
    )
    assert all(
        mesh_origins[link].shape == (4, 4) for link in mesh_origins
    ), "Expected a 4x4 shape for each mesh origin."
    for link in mesh_origins:
        print(mesh_origins[link][3, :])
    assert all(
        (mesh_origins[link][3, :] == [0.0, 0.0, 0.0, 1.0]).all()
        for link in mesh_origins
    ), "Expected homogeneous transforms for mesh origins."


@pytest.mark.skip(reason="To be fixed.")
def test_urdf_parser_from_ros_xacro() -> None:
    urdf_parser = URDFParser.from_ros_xacro("lbr_description", "urdf/med7/med7.xacro")
    print(urdf_parser.chain_link_names("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.mesh_uris("lbr_link_0", "lbr_link_ee"))
    print(
        URDFParser.resolve_ros_registry_uris(
            urdf_parser.mesh_uris("lbr_link_0", "lbr_link_ee")
        )
    )
    print(urdf_parser.mesh_paths_from_ros_registry("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.mesh_origins("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.link_names_with_meshes(collision=True))
    print(urdf_parser.link_names_with_meshes(collision=False))


def test_find_files() -> None:
    path = "test/assets/lbr_med7_r800/samples"
    mask_files = find_files(path, "mask_sam2_left_*.png")

    assert len(mask_files) > 0, "Should find at least one mask file."
    assert all(
        isinstance(f, Path) for f in mask_files
    ), "All results should be Path objects."
    assert all(f.exists() for f in mask_files), "All files should exist."
    assert all(f.suffix == ".png" for f in mask_files), "All files should be .png."


def test_parse_camera_info() -> None:
    path = Path("test/assets/lbr_med7_r800/samples")
    file = "left_camera_info.yaml"
    height, width, intrinsic_matrix = parse_camera_info(path / file)

    assert isinstance(height, int), "Height should be an integer."
    assert isinstance(width, int), "Width should be an integer."
    assert height > 0, "Height should be positive."
    assert width > 0, "Width should be positive."
    assert isinstance(intrinsic_matrix, np.ndarray)
    assert intrinsic_matrix.shape == (3, 3), "Intrinsic matrix should be of shape 3x3."


def test_parse_hydra_observations() -> None:
    path = "test/assets/lbr_med7_r800/samples"
    observations = parse_hydra_observations(
        joint_states_files=find_files(path, "joint_states_*.npy"),
        mask_files=find_files(path, "mask_sam2_left_*.png"),
        depth_files=find_files(path, "depth_*.npy"),
    )

    assert (
        len(observations.joint_states)
        == len(observations.masks)
        == len(observations.depths)
    ), "Expected same number of joint states / masks / depths."
    assert len(observations.joint_states) >= 1, "Should at least have one sample."
    assert observations.masks[0].ndim == 2, "Expected 2D mask."
    assert (
        observations.masks[0].dtype == np.uint8
    ), "Expected unsigned integers for mask."
    assert np.all(observations.masks[0] >= 0) and np.all(
        observations.masks[0] <= 255
    ), "Expected mask in range [0, 255]."
    assert observations.depths[0].ndim == 2, "Expected 2D depth map."


def test_parse_monocular_observations() -> None:
    path = "test/assets/lbr_med7_r800/samples"
    observations = parse_monocular_observations(
        image_files=find_files(path, "left_image_*.png"),
        joint_states_files=find_files(path, "joint_states_*.npy"),
        target_files=find_files(path, "mask_sam2_left_*.png"),
    )

    assert (
        len(observations.images)
        == len(observations.joint_states)
        == len(observations.targets)
    ), "Expected same number of images / joint states / masks."
    assert len(observations.images) >= 1, "Should at least have one sample."
    assert observations.images[0].ndim == 3, "Expected 3D image (HxWx3)."
    assert observations.images[0].shape[-1] == 3, "Expected 3 color channels."
    assert observations.targets[0].ndim == 2, "Expected 2D mask."
    assert (
        observations.targets[0].dtype == np.uint8
    ), "Expected unsigned integers for mask."
    assert np.all(observations.targets[0] >= 0) and np.all(
        observations.targets[0] <= 255
    ), "Expected mask in range [0, 255]."
    assert (
        observations.targets[0].shape[:2] == observations.images[0].shape[:2]
    ), "Mask and image dimensions should match."


def test_parse_stereo_observations() -> None:
    path = "test/assets/lbr_med7_r800/samples"
    observations = parse_stereo_observations(
        left_image_files=find_files(path, "left_image_*.png"),
        right_image_files=find_files(path, "right_image_*.png"),
        joint_states_files=find_files(path, "joint_states_*.npy"),
        left_target_files=find_files(path, "mask_sam2_left_*.png"),
        right_target_files=find_files(path, "mask_sam2_right_*.png"),
    )

    assert (
        len(observations.left_images)
        == len(observations.right_images)
        == len(observations.joint_states)
        == len(observations.left_targets)
        == len(observations.right_targets)
    ), "Expected same number of left/right images, joint states, and left/right masks."
    assert len(observations.left_images) >= 1, "Should at least have one sample."

    # Test left data
    assert observations.left_images[0].ndim == 3, "Expected 3D left image (HxWx3)."
    assert (
        observations.left_images[0].shape[-1] == 3
    ), "Expected 3 color channels for left image."
    assert observations.left_targets[0].ndim == 2, "Expected 2D left mask."
    assert (
        observations.left_targets[0].dtype == np.uint8
    ), "Expected unsigned integers for left mask."
    assert np.all(observations.left_targets[0] >= 0) and np.all(
        observations.left_targets[0] <= 255
    ), "Expected left mask in range [0, 255]."

    # Test right data
    assert observations.right_images[0].ndim == 3, "Expected 3D right image (HxWx3)."
    assert (
        observations.right_images[0].shape[-1] == 3
    ), "Expected 3 color channels for right image."
    assert observations.right_targets[0].ndim == 2, "Expected 2D right mask."
    assert (
        observations.right_targets[0].dtype == np.uint8
    ), "Expected unsigned integers for right mask."
    assert np.all(observations.right_targets[0] >= 0) and np.all(
        observations.right_targets[0] <= 255
    ), "Expected right mask in range [0, 255]."

    # Test dimensions match
    assert (
        observations.left_targets[0].shape[:2] == observations.left_images[0].shape[:2]
    ), "Left mask and image dimensions should match."
    assert (
        observations.right_targets[0].shape[:2]
        == observations.right_images[0].shape[:2]
    ), "Right mask and image dimensions should match."


if __name__ == "__main__":
    import os
    import sys

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_urdf_parser_from_urdf_file()
    test_urdf_parser_from_ros_xacro()
    test_find_files()
    test_parse_camera_info()
    test_parse_hydra_observations()
    test_parse_monocular_observations()
    test_parse_stereo_observations()
