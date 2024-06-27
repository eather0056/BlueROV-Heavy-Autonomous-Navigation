cmake_minimum_required(VERSION 3.5)
project(webcam_publisher)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# Python executables
install(PROGRAMS
  src/webcam_publisher_node.py 
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(rclpy sensor_msgs cv_bridge)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
