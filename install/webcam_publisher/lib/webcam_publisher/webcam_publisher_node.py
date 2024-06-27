#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamPublisherNode(Node):
    def __init__(self):
        super().__init__('webcam_publisher_node')
        self.publisher = self.create_publisher(Image, '/video_frames', 10)
        self.cap = cv2.VideoCapture(4)  # Open default camera (usually webcam)
        self.bridge = CvBridge()

        self.timer = self.create_timer(0.1, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(msg)

    def shutdown(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
