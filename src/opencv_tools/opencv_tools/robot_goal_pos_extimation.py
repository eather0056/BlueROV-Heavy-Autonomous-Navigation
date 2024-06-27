#!/usr/bin/env python3

from __future__ import print_function
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import sys

# Dictionary that was used to generate the ArUco marker
aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"

# The different ArUco dictionaries built into the OpenCV library.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# Side length of the ArUco marker in meters
aruco_marker_side_length = 0.13

# Calibration parameters yaml file
camera_calibration_parameters_filename = '/home/eth/ros2_ws/src/opencv_tools/opencv_tools/calibration_chessboard.yaml'

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def calculate_robot_position(marker_positions):
    """
    Calculate the robot's position based on detected markers 1 and 2.
    """
    if 1 in marker_positions and 2 in marker_positions:
        x1, y1, z1 = marker_positions[1]
        x2, y2, z2 = marker_positions[2]
        robot_x = (x1 + x2) / 2
        robot_y = (y1 + y2) / 2
        robot_z = (z1 + z2) / 2
    elif 1 in marker_positions:
        robot_x, robot_y, robot_z = marker_positions[1]
    elif 2 in marker_positions:
        robot_x, robot_y, robot_z = marker_positions[2]
    else:
        robot_x, robot_y, robot_z = None, None, None
    
    return robot_x, robot_y, robot_z

def calculate_distance_and_orientation(robot_pos, goal_pos, robot_yaw, goal_yaw):
    """
    Calculate the distance and orientation difference between the robot and the goal.
    """
    if robot_pos is None or goal_pos is None:
        return None, None
    
    distance = np.linalg.norm(np.array(robot_pos) - np.array(goal_pos))
    
    # Calculate yaw difference in degrees (0 to 360)
    yaw_diff = math.degrees(robot_yaw - goal_yaw) % 360
    if yaw_diff < 0:
        yaw_diff += 360
    
    return distance, yaw_diff

class ArucoNode(Node):
    """
    Create an ArucoNode class, which is a subclass of the Node class.
    """
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('aruco_node')

        # Declare parameters
        self.declare_parameter("aruco_dictionary_name", "DICT_ARUCO_ORIGINAL")
        self.declare_parameter("aruco_marker_side_length", 0.13)
        self.declare_parameter("camera_calibration_parameters_filename", "/home/eth/ros2_ws/src/opencv_tools/opencv_tools/calibration_chessboard.yaml")
        self.declare_parameter("image_topic", "/video_frames")
        self.declare_parameter("aruco_robot_marker_name", "aruco_robot")
        self.declare_parameter("aruco_goal_marker_name", "aruco_goal")

        # Read parameters
        aruco_dictionary_name = self.get_parameter("aruco_dictionary_name").get_parameter_value().string_value
        self.aruco_marker_side_length = self.get_parameter("aruco_marker_side_length").get_parameter_value().double_value
        self.camera_calibration_parameters_filename = self.get_parameter(
            "camera_calibration_parameters_filename").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.aruco_robot_marker_name = self.get_parameter("aruco_robot_marker_name").get_parameter_value().string_value
        self.aruco_goal_marker_name = self.get_parameter("aruco_goal_marker_name").get_parameter_value().string_value

        # Check that we have a valid ArUco marker
        if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
            self.get_logger().info("[INFO] ArUCo tag of '{}' is not supported".format(
                aruco_dictionary_name))

        # Load the camera parameters from the saved file
        cv_file = cv2.FileStorage(
            self.camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
        self.mtx = cv_file.getNode('K').mat()
        self.dst = cv_file.getNode('D').mat()
        cv_file.release()

        # Load the ArUco dictionary
        self.get_logger().info("[INFO] detecting '{}' markers...".format(
            aruco_dictionary_name))
        self.this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_dictionary_name])
        self.this_aruco_parameters = cv2.aruco.DetectorParameters_create()

        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
            Image, 
            image_topic, 
            self.listener_callback, 
            10)
        self.subscription # prevent unused variable warning

        # Initialize the transform broadcaster
        self.tfbroadcaster = TransformBroadcaster(self)

        # Used to convert between ROS and OpenCV images
        self.bridge = CvBridge()

    def listener_callback(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        self.get_logger().info('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        current_frame = self.bridge.imgmsg_to_cv2(data)

        # Detect ArUco markers in the video frame
        (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
            current_frame, self.this_aruco_dictionary, parameters=self.this_aruco_parameters,
            cameraMatrix=self.mtx, distCoeff=self.dst)

        marker_positions = {}
        robot_yaw = None
        goal_yaw = None

        # Draw a square around the markers
        if marker_ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.aruco_marker_side_length, self.mtx, self.dst)

            for i, marker_id in enumerate(marker_ids):
                marker_id = marker_id[0]
                cv2.aruco.drawDetectedMarkers(current_frame, corners, marker_ids)
                cv2.aruco.drawAxis(current_frame, self.mtx, self.dst, rvecs[i], tvecs[i], 0.1)

                # Store the marker position
                marker_positions[marker_id] = tvecs[i][0]

                # Convert the rotation vector to a rotation matrix
                rmat = R.from_rotvec(rvecs[i][0]).as_matrix()

                # Create a transform message for the marker
                transform = TransformStamped()
                transform.header.stamp = self.get_clock().now().to_msg()
                transform.header.frame_id = 'camera_frame'
                transform.child_frame_id = 'aruco_marker_' + str(marker_id)
                transform.transform.translation.x = tvecs[i][0][0]
                transform.transform.translation.y = tvecs[i][0][1]
                transform.transform.translation.z = tvecs[i][0][2]

                # Get the quaternion from the rotation matrix
                quat = R.from_matrix(rmat).as_quat()
                transform.transform.rotation.x = quat[0]
                transform.transform.rotation.y = quat[1]
                transform.transform.rotation.z = quat[2]
                transform.transform.rotation.w = quat[3]

                # Send the transform
                self.tfbroadcaster.sendTransform(transform)

                # Save the yaw for robot and goal markers
                if marker_id == 1:
                    robot_yaw = euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])[2]
                elif marker_id == 2:
                    goal_yaw = euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])[2]

        # Calculate robot position
        robot_pos = calculate_robot_position(marker_positions)
        goal_pos = marker_positions.get(2, None)

        # Calculate distance and orientation
        distance, orientation = calculate_distance_and_orientation(robot_pos, goal_pos, robot_yaw, goal_yaw)

        if distance is not None and orientation is not None:
            self.get_logger().info(f"Distance to goal: {distance:.2f} meters")
            self.get_logger().info(f"Orientation difference: {orientation:.2f} degrees")

        # Display the resulting frame
        cv2.imshow('Aruco Marker Detection', current_frame)
        cv2.waitKey(1)

def main(args=None):
    """
    Entry point for the program.
    """
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    aruco_node = ArucoNode()

    # Spin the node so the callback function is called.
    rclpy.spin(aruco_node)

    # Destroy the node explicitly
    aruco_node.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
