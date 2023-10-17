import os
import pathlib
from typing import Tuple

import cv2
import cv_bridge
import numpy as np
import rosbag2_py
from common import get_rosbag_options
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, JointState, PointCloud2

bridge = cv_bridge.CvBridge()


def image_to_numpy(image: Image) -> np.ndarray:
    np_image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
    return np_image


def joint_state_to_numpy(
    joint_state: JointState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # sort arrays by name
    position = joint_state.position
    velocity = joint_state.velocity
    effort = joint_state.effort
    name = joint_state.name
    position = [x for _, x in sorted(zip(name, position))]
    velocity = [x for _, x in sorted(zip(name, velocity))]
    effort = [x for _, x in sorted(zip(name, effort))]
    return (
        np.array(position),
        np.array(velocity),
        np.array(effort),
    )


def point_cloud_to_numpy(point_cloud: PointCloud2) -> np.ndarray:
    data = np.array(point_cloud.data, dtype=np.uint8)

    # offset + byte, step size = 4 * 4 bytes
    x_b0, x_b1, x_b2, x_b3 = (
        data[point_cloud.fields[0].offset + 0 :: point_cloud.point_step],
        data[point_cloud.fields[0].offset + 1 :: point_cloud.point_step],
        data[point_cloud.fields[0].offset + 2 :: point_cloud.point_step],
        data[point_cloud.fields[0].offset + 3 :: point_cloud.point_step],
    )
    y_b0, y_b1, y_b2, y_b3 = (
        data[point_cloud.fields[1].offset + 0 :: point_cloud.point_step],
        data[point_cloud.fields[1].offset + 1 :: point_cloud.point_step],
        data[point_cloud.fields[1].offset + 2 :: point_cloud.point_step],
        data[point_cloud.fields[1].offset + 3 :: point_cloud.point_step],
    )
    z_b0, z_b1, z_b2, z_b3 = (
        data[point_cloud.fields[2].offset + 0 :: point_cloud.point_step],
        data[point_cloud.fields[2].offset + 1 :: point_cloud.point_step],
        data[point_cloud.fields[2].offset + 2 :: point_cloud.point_step],
        data[point_cloud.fields[2].offset + 3 :: point_cloud.point_step],
    )
    rgb_b0, rgb_b1, rgb_b2, rgb_b3 = (
        data[point_cloud.fields[3].offset + 0 :: point_cloud.point_step],
        data[point_cloud.fields[3].offset + 1 :: point_cloud.point_step],
        data[point_cloud.fields[3].offset + 2 :: point_cloud.point_step],
        data[point_cloud.fields[3].offset + 3 :: point_cloud.point_step],
    )

    x = np.stack([x_b0, x_b1, x_b2, x_b3], axis=1)
    y = np.stack([y_b0, y_b1, y_b2, y_b3], axis=1)
    z = np.stack([z_b0, z_b1, z_b2, z_b3], axis=1)
    rgba = np.stack([rgb_b0, rgb_b1, rgb_b2, rgb_b3], axis=1)

    height, width = point_cloud.height, point_cloud.width

    x = x.flatten().view(dtype=np.float32).reshape((height, width))
    y = y.flatten().view(dtype=np.float32).reshape((height, width))
    z = z.flatten().view(dtype=np.float32).reshape((height, width))
    rgba = rgba.reshape((height, width, 4))

    return x, y, z, rgba


def visualize_point_cloud(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rgba: np.ndarray
):
    # normalize for visualization (as in units of meters)
    x[~np.isfinite(x)] = 0.0
    y[~np.isfinite(y)] = 0.0
    z[~np.isfinite(z)] = 0.0

    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    z = (z - np.min(z)) / (np.max(z) - np.min(z))

    # color convert
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

    cv2.imshow("x", x)
    cv2.imshow("y", y)
    cv2.imshow("z", z)
    cv2.imshow("rgb", rgb)


def visualize_image(image: np.ndarray):
    cv2.imshow("image", image)


def process_rosbag(bag_path, topics, absolute_output_path, visualize=False):
    absolute_output_path = pathlib.Path(absolute_output_path)
    if not absolute_output_path.exists():
        absolute_output_path.mkdir(parents=True)

    kinematics_absolute_output_path = pathlib.Path(absolute_output_path / "kinematics")
    if not kinematics_absolute_output_path.exists():
        kinematics_absolute_output_path.mkdir(parents=True)

    camera_absolute_output_path = pathlib.Path(absolute_output_path / "camera")
    if not camera_absolute_output_path.exists():
        camera_absolute_output_path.mkdir(parents=True)

    point_cloud_absolute_output_path = pathlib.Path(
        absolute_output_path / "point_cloud"
    )
    if not point_cloud_absolute_output_path.exists():
        point_cloud_absolute_output_path.mkdir(parents=True)

    storage_options, converter_options = get_rosbag_options(bag_path)
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()

    # Create a map for quicker lookup
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    # Set filter for topic of string type
    storage_filter = rosbag2_py.StorageFilter(topics=topics)
    reader.set_filter(storage_filter)

    last_image = Image()
    last_joint_state = JointState()
    last_point_cloud = PointCloud2()

    save_count = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if msg_type == Image:
            last_image = msg
        if msg_type == JointState:
            last_joint_state = msg
        if msg_type == PointCloud2:
            last_point_cloud = msg

            if last_image.height == 0:
                continue
            if len(last_joint_state.position) == 0:
                continue

            img = image_to_numpy(last_image)
            position, _, _ = joint_state_to_numpy(last_joint_state)
            x, y, z, rgba = point_cloud_to_numpy(last_point_cloud)

            np.save(
                os.path.join(
                    camera_absolute_output_path.absolute(), f"img_{save_count}.npy"
                ),
                img,
            )
            np.save(
                os.path.join(
                    kinematics_absolute_output_path.absolute(),
                    f"position_{save_count}.npy",
                ),
                position,
            )
            np.save(
                os.path.join(
                    point_cloud_absolute_output_path.absolute(),
                    f"x_{save_count}.npy",
                ),
                x,
            )
            np.save(
                os.path.join(
                    point_cloud_absolute_output_path.absolute(),
                    f"y_{save_count}.npy",
                ),
                y,
            )
            np.save(
                os.path.join(
                    point_cloud_absolute_output_path.absolute(),
                    f"z_{save_count}.npy",
                ),
                z,
            )
            np.save(
                os.path.join(
                    point_cloud_absolute_output_path.absolute(),
                    f"rgba_{save_count}.npy",
                ),
                rgba,
            )

            save_count += 1

            if visualize:
                visualize_point_cloud(x, y, z, rgba)
                visualize_image(img)
                cv2.waitKey()


if __name__ == "__main__":
    bag_path = "/media/martin/Samsung_T5/23_07_04_faros_integration_week_measurements/self_registration/self_observation_rosbag"
    topics = [
        "/zed2/zed_node/point_cloud/cloud_registered",
        "/joint_states",
        "/zed2/zed_node/left/image_rect_color",
    ]
    process_rosbag(
        bag_path,
        topics,
        absolute_output_path="/media/martin/Samsung_T5/23_07_04_faros_integration_week_measurements/self_registration/self_observation_rosbag",
        visualize=True,
    )
