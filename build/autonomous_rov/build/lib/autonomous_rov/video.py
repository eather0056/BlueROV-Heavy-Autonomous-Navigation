#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import gi
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

gi.require_version('Gst', '1.0')
from gi.repository import Gst

class Controller(Node):
    """BlueRov video capture class constructor"""

    g = 9.81  # m.s^-2 gravitational acceleration
    p0 = 103425  # Surface pressure in Pascal
    rho = 1000  # kg/m^3 water density

    def __init__(self):
        super().__init__("video")

        self.declare_parameter("port", 5600)

        self.port = self.get_parameter("port").value
        self._frame = None
        self.video_source = 'udpsrc port={}'.format(self.port)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        self.video_decode = '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        self.video_sink_conf = '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        # font
        self.font = cv2.FONT_HERSHEY_PLAIN

        Gst.init()

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, 'video_frames', 10)

        self.run()

        # Start update loop
        self.create_timer(0.01, self.update)

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink """
        if not config:
            config = [
                'videotestsrc ! decodebin',
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array"""
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame """
        return self._frame

    def frame_available(self):
        """Check if frame is available"""
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame """
        self.start_gst([
            self.video_source,
            self.video_codec,
            self.video_decode,
            self.video_sink_conf
        ])
        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame
        return Gst.FlowReturn.OK

    def update(self):
        if not self.frame_available():
            return

        frame = self.frame()
        width = int(1920 / 1.5)
        height = int(1080 / 1.5)
        dim = (width, height)
        img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        self.draw_gui(img, width, height)

        cv2.imshow('BlueROV2 Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()

        # Convert OpenCV image to ROS2 Image message
        image_message = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.publisher.publish(image_message)

    def draw_gui(self, img, width, height):
        img = cv2.rectangle(img, (0, height - 100), (520, height), (0, 0, 0), -1)

def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
