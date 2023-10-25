import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

if __name__ == "__main__":
    rclpy.init()

    node = Node("transform_provider")

    buffer = Buffer()
    listener = TransformListener(buffer, node)

    while rclpy.ok():
        rclpy.spin_once(node)
        try:
            tf = buffer.lookup_transform("zed2_left_camera_frame", "world", Time())
            print(tf)
            break
        except:
            pass

    rclpy.shutdown()
