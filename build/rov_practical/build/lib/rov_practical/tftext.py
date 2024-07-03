import rclpy
from rclpy.node import Node
import tf_transformations as tf
import math

class TFExample(Node):
    def __init__(self):
        super().__init__('tf_example')
        # Example quaternion
        quaternion = (0.0, 0.0, 0.383, 0.924)

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = tf.euler_from_quaternion(quaternion)

        # Print the Euler angles in degrees
        self.get_logger().info(f"Quaternion: {quaternion}")
        self.get_logger().info(f"Euler Angles (in degrees): Roll={math.degrees(roll)}, Pitch={math.degrees(pitch)}, Yaw={math.degrees(yaw)}")

        # Example Euler angles in radians
        roll, pitch, yaw = 0.785, 0.524, 1.047

        # Convert Euler angles to quaternion
        quaternion = tf.quaternion_from_euler(roll, pitch, yaw)

        self.get_logger().info(f"Euler Angles (in radians): Roll={roll}, Pitch={pitch}, Yaw={yaw}")
        self.get_logger().info(f"Quaternion: {quaternion}")

def main(args=None):
    rclpy.init(args=args)
    node = TFExample()
    rclpy.spin(node)
    rclpy.shutdown()

