from pathlib import Path

import numpy as np
import pytest

from roboreg.io import (
    URDFParser,
    find_files,
    parse_camera_info,
    parse_hydra_data,
    parse_mono_data,
    parse_stereo_data,
)


@pytest.mark.skip(reason="To be fixed.")
def test_urdf_parser() -> None:
    urdf_parser = URDFParser.from_ros_xacro("lbr_description", "urdf/med7/med7.xacro")
    print(urdf_parser.chain_link_names("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.raw_mesh_paths("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.mesh_paths_from_ros_registry("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.mesh_origins("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.link_names_with_meshes(collision=True))
    print(urdf_parser.link_names_with_meshes(collision=False))


def test_find_files() -> None:
    path = "test/assets/lbr_med7/zed2i"
    mask_files = find_files(path, "mask_sam2_left_*.png")

    assert len(mask_files) > 0, "Should find at least one mask file."
    assert all(
        isinstance(f, Path) for f in mask_files
    ), "All results should be Path objects."
    assert all(f.exists() for f in mask_files), "All files should exist."
    assert all(f.suffix == ".png" for f in mask_files), "All files should be .png."


def test_parse_camera_info() -> None:
    path = Path("test/assets/lbr_med7/zed2i")
    file = "left_camera_info.yaml"
    height, width, intrinsic_matrix = parse_camera_info(path / file)

    assert isinstance(height, int), "Height should be an integer."
    assert isinstance(width, int), "Width should be an integer."
    assert height > 0, "Height should be positive."
    assert width > 0, "Width should be positive."
    assert isinstance(intrinsic_matrix, np.ndarray)
    assert intrinsic_matrix.shape == (3, 3), "Intrinsic matrix should be of shape 3x3."


def test_parse_hydra_data() -> None:
    path = "test/assets/lbr_med7/zed2i"
    joint_states, masks, depths = parse_hydra_data(
        joint_states_files=find_files(path, "joint_states_*.npy"),
        mask_files=find_files(path, "mask_sam2_left_*.png"),
        depth_files=find_files(path, "depth_*.npy"),
    )

    assert (
        len(joint_states) == len(masks) == len(depths)
    ), "Expected same number of joint states / masks / depths."
    assert len(joint_states) >= 1, "Should at least have one sample."
    assert masks[0].ndim == 2, "Expected 2D mask."
    assert masks[0].dtype == np.uint8, "Expected unsigned integers for mask."
    assert np.all(masks[0] >= 0) and np.all(
        masks[0] <= 255
    ), "Expected mask in range [0, 255]."
    assert depths[0].ndim == 2, "Expected 2D depth map."


def test_parse_mono_data() -> None:
    path = "test/assets/lbr_med7/zed2i"
    images, joint_states, masks = parse_mono_data(
        image_files=find_files(path, "left_image_*.png"),
        joint_states_files=find_files(path, "joint_states_*.npy"),
        mask_files=find_files(path, "mask_sam2_left_*.png"),
    )

    assert (
        len(images) == len(joint_states) == len(masks)
    ), "Expected same number of images / joint states / masks."
    assert len(images) >= 1, "Should at least have one sample."
    assert images[0].ndim == 3, "Expected 3D image (HxWx3)."
    assert images[0].shape[-1] == 3, "Expected 3 color channels."
    assert masks[0].ndim == 2, "Expected 2D mask."
    assert masks[0].dtype == np.uint8, "Expected unsigned integers for mask."
    assert np.all(masks[0] >= 0) and np.all(
        masks[0] <= 255
    ), "Expected mask in range [0, 255]."
    assert (
        masks[0].shape[:2] == images[0].shape[:2]
    ), "Mask and image dimensions should match."


def test_parse_stereo_data() -> None:
    path = "test/assets/lbr_med7/zed2i"
    left_images, right_images, joint_states, left_masks, right_masks = (
        parse_stereo_data(
            left_image_files=find_files(path, "left_image_*.png"),
            right_image_files=find_files(path, "right_image_*.png"),
            joint_states_files=find_files(path, "joint_states_*.npy"),
            left_mask_files=find_files(path, "mask_sam2_left_*.png"),
            right_mask_files=find_files(path, "mask_sam2_right_*.png"),
        )
    )

    assert (
        len(left_images)
        == len(right_images)
        == len(joint_states)
        == len(left_masks)
        == len(right_masks)
    ), "Expected same number of left/right images, joint states, and left/right masks."
    assert len(left_images) >= 1, "Should at least have one sample."

    # Test left data
    assert left_images[0].ndim == 3, "Expected 3D left image (HxWx3)."
    assert left_images[0].shape[-1] == 3, "Expected 3 color channels for left image."
    assert left_masks[0].ndim == 2, "Expected 2D left mask."
    assert left_masks[0].dtype == np.uint8, "Expected unsigned integers for left mask."
    assert np.all(left_masks[0] >= 0) and np.all(
        left_masks[0] <= 255
    ), "Expected left mask in range [0, 255]."

    # Test right data
    assert right_images[0].ndim == 3, "Expected 3D right image (HxWx3)."
    assert right_images[0].shape[-1] == 3, "Expected 3 color channels for right image."
    assert right_masks[0].ndim == 2, "Expected 2D right mask."
    assert (
        right_masks[0].dtype == np.uint8
    ), "Expected unsigned integers for right mask."
    assert np.all(right_masks[0] >= 0) and np.all(
        right_masks[0] <= 255
    ), "Expected right mask in range [0, 255]."

    # Test dimensions match
    assert (
        left_masks[0].shape[:2] == left_images[0].shape[:2]
    ), "Left mask and image dimensions should match."
    assert (
        right_masks[0].shape[:2] == right_images[0].shape[:2]
    ), "Right mask and image dimensions should match."


if __name__ == "__main__":
    import os
    import sys

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_urdf_parser()
    test_find_files()
    test_parse_camera_info()
    test_parse_hydra_data()
    test_parse_mono_data()
    test_parse_stereo_data()
