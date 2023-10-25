import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster, TransformListener
from tf2_ros.buffer import Buffer

from scipy.spatial import transform

# handle static transform


class TFConversion(Node):
    def __init__(self, node_name: str, tf: Transform):
        super().__init__(node_name=node_name)
        self.tf = tf
        self.tf_static_sub = self.create_subscription(
            TFMessage,
            "/old/tf_static",
            self.on_tf_static,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.tf_static_pub = self.create_publisher(
            TFMessage,
            "/tf_static",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

    def on_tf_static(self, tf_msg: TFMessage) -> None:
        for tf in tf_msg.transforms:
            self.get_logger().info(
                f"frame: {tf.header.frame_id}, child: {tf.child_frame_id}"
            )
        tf_msg.transforms.append(  # somehow missing?? anyways, change it
            TransformStamped(
                header=Header(
                    stamp=tf_msg.transforms[0].header.stamp, frame_id="world"
                ),
                child_frame_id="lbr_link_0",
                transform=self.tf,
            )
        )
        tf_msg.transforms.append(  # somehow missing?? anyways, change it
            TransformStamped(
                header=Header(stamp=tf_msg.transforms[0].header.stamp, frame_id="map"),
                child_frame_id="world",
                transform=Transform(),
            )
        )
        self.tf_static_pub.publish(tf_msg)


if __name__ == "__main__":
    ## requires remapping of rosbag from /tf_static to /old/tf_static
    ht_world_to_base = np.load("registration/homogeneous_transform.npy")
    ht_world_to_base = np.linalg.inv(ht_world_to_base)
    print(f"ht: {ht_world_to_base}")

    # TODO: add transform zed2_left_camera_frame -> world to the world to lbr_link_0 transform
    # obtained from zed_camera_frame_to_world.py
    ht_zed_to_world = np.eye(4)
    ht_zed_to_world[:3, :3] = transform.Rotation(
        [
            0.00017263348523432538,
            -0.02501964893623179,
            0.00012207031015820437,
            0.9996869372276638,
        ]
    ).as_matrix()
    ht_zed_to_world[:3, 3] = [
        0.0002730033584825046,
        -0.06005021130180059,
        -0.014860413692497031,
    ]

    ht = np.linalg.inv(ht_zed_to_world) @ ht_world_to_base

    rotation = transform.Rotation.from_matrix(ht[:3, :3]).as_quat()
    print(f"rotation: {rotation}")

    tf = Transform(
        translation=Vector3(x=ht[0, 3], y=ht[1, 3], z=ht[2, 3]),
        rotation=Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3]),
    )
    # tf = Transform()
    print(tf)

    rclpy.init()
    node = TFConversion("broadcaster", tf)
    rclpy.spin(node)
    rclpy.shutdown()
