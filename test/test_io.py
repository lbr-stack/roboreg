import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro("lbr_description", "urdf/med7/med7.xacro")
    print(urdf_parser.chain_link_names("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.raw_mesh_paths("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.ros_package_mesh_paths("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.mesh_origins("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.link_names_with_meshes(collision=True))
    print(urdf_parser.link_names_with_meshes(collision=False))


@pytest.mark.skip(reason="To be fixed.")
def test_find_files() -> None:
    path = "test/assets/lbr_med7/zed2i"
    for mask_file in find_files(path, "mask_sam2_left_*.png"):
        print(mask_file)


@pytest.mark.skip(reason="To be fixed.")
def test_parse_camera_info() -> None:
    path = "test/assets/lbr_med7/zed2i"
    file = "left_camera_info.yaml"
    height, width, intrinsic_matrix = parse_camera_info(os.path.join(path, file))

    print(height)
    print(width)
    print(intrinsic_matrix)


@pytest.mark.skip(reason="To be fixed.")
def test_parse_hydra_data() -> None:
    path = "test/assets/lbr_med7/zed2i"
    joint_states_files = find_files(path, "joint_states_*.npy")
    mask_files = find_files(path, "mask_sam2_left_*.png")
    depth_files = find_files(path, "depth_*.npy")
    joint_states, masks, depths = parse_hydra_data(
        path,
        joint_states_files=joint_states_files,
        mask_files=mask_files,
        depth_files=depth_files,
    )
    print(len(joint_states))
    print(len(masks))
    print(len(depths))
    print(joint_states[0].shape)
    print(masks[0].shape)
    print(depths[0].shape)


@pytest.mark.skip(reason="To be fixed.")
def test_parse_mono_data() -> None:
    path = "test/assets/lbr_med7/zed2i"
    image_files = find_files(path, "left_image_*.png")
    joint_states_files = find_files(path, "joint_states_*.npy")
    mask_files = find_files(path, "mask_sam2_left_*.png")
    images, joint_states, masks = parse_mono_data(
        path,
        image_files=image_files,
        joint_states_files=joint_states_files,
        mask_files=mask_files,
    )
    print(len(images))
    print(len(joint_states))
    print(len(masks))
    print(images[0].shape)
    print(joint_states[0].shape)
    print(masks[0].shape)


@pytest.mark.skip(reason="To be fixed.")
def test_parse_stereo_data() -> None:
    path = "test/assets/lbr_med7/zed2i"
    left_image_files = find_files(path, "left_image_*.png")
    right_image_files = find_files(path, "right_image_*.png")
    joint_states_files = find_files(path, "joint_states_*.npy")
    left_mask_files = find_files(path, "mask_sam2_left_*.png")
    right_mask_files = find_files(path, "mask_sam2_right_*.png")
    left_images, right_images, joint_states, left_masks, right_masks = (
        parse_stereo_data(
            "test/assets/lbr_med7/zed2i",
            left_image_files=left_image_files,
            right_image_files=right_image_files,
            joint_states_files=joint_states_files,
            left_mask_files=left_mask_files,
            right_mask_files=right_mask_files,
        )
    )
    print(len(left_images))
    print(len(right_images))
    print(len(joint_states))
    print(len(left_masks))
    print(len(right_masks))
    print(left_images[0].shape)
    print(right_images[0].shape)
    print(joint_states[0].shape)
    print(left_masks[0].shape)
    print(right_masks[0].shape)


if __name__ == "__main__":
    test_urdf_parser()
    test_find_files()
    test_parse_camera_info()
    test_parse_hydra_data()
    test_parse_mono_data()
    test_parse_stereo_data()
