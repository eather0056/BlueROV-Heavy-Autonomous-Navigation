# ROS2 Workspace for BlueROV Heavy Autonomous Navigation

This repository contains ROS2 packages for the development and deployment of an autonomous remotely operated vehicle (ROV), specifically the BlueROV Heavy. The workspace is organized into several packages, including `autonomous_rov`, `opencv_tools`, and `rov_practical`.

## Repository Structure

```
ros2_ws/
├── src/
│   ├── autonomous_rov/
│   │   ├── autonomous_rov/
│   │   ├── config/
│   │   ├── launch/
│   │   ├── package.xml
│   │   ├── resource/
│   │   ├── setup.cfg
│   │   ├── setup.py
│   │   └── test/
│   ├── opencv_tools/
│   │   ├── opencv_tools/
│   │   ├── package.xml
│   │   ├── resource/
│   │   ├── setup.cfg
│   │   ├── setup.py
│   │   └── test/
│   └── rov_practical/
│       ├── package.xml
│       ├── resource/
│       ├── rov_practical/
│       ├── setup.cfg
│       ├── setup.py
│       └── test/
└── README.md
```

## Packages Overview

### `autonomous_rov`
This package is responsible for controlling the ROV and setting up the BlueROV Navigator. Detailed instructions for setup and usage can be found in the [MRI-BlueROV-Heavy-Setup](https://github.com/eather0056/MRI-BlueROV-Heavy-Setup.git) repository.

### `opencv_tools`
The `opencv_tools` package contains the localization method of the robot agent and the goal position in a tank environment using Aruco markers. The position of the agent and goal will be presented in the camera frame and can be transferred to the world frame. This information is published as a tf and can be observed in the Rviz2 tool. Detailed instructions for setup and usage can be found in the [Aruco Marker Pose Estimation](https://github.com/eather0056/Aruco-Marker-Pose-Estimation.git) repository.

### `rov_practical`
The `rov_practical` package implements the DDDQN method for training and testing the agent using different Reinforcement Learning (RL) algorithms. It utilizes `NavSatFix`, `Imu`, `Image`, and `LaserScan` sensor data to interact with the environment. Detailed instructions for setup and implementation can be found in the [BlueROV_Navigation_Control_with_Reinforcement_Learning](https://github.com/eather0056/BlueROV_Navigation_Control_with_Reinforcement_Learning.git) repository.

## Prerequisites

Make sure you have the following installed:

- [ROS2 Humble Hawksbll](https://docs.ros.org/en/humble/Installation.html), any other ROS2 Release
- Python 3.8 or later
- [Colcon](https://colcon.readthedocs.io/en/released/)

## Installation

### Clone the Repository

```bash
git clone https://github.com/eather0056/BlueROV-Heavy-Autonomous-Navigation.git
cd BlueROV-Heavy-Autonomous-Navigation/ros2_ws
```

### Install kobuki_ros_interfaces
```bash
cd src
git clone https://github.com/kobuki-base/kobuki_ros_interfaces.git
```

### Install Dependencies

Use the provided `requirements.txt` file to install necessary Python packages:

```bash
cd ~/ros2_ws
pip install -r requirements.txt
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### Installing `geometry_msgs`
```bash
sudo apt-get update
sudo apt-get install ros-humble-geometry-msgs
```

### Build the Workspace

```bash
source /opt/ros/humble/setup.bash
colcon build
```

## Usage

### Source the Workspace

Before running any ROS2 nodes, source the workspace:

```bash
source install/setup.bash
```

### Running the Nodes

### Verify the kobuki_ros_interfaces Installation

```bash
ros2 interface show kobuki_ros_interfaces/msg/BumperEvent
```

#### Autonomous ROV

Navigate to the `launch` directory of the `autonomous_rov` package to launch the ROV:

```bash
ros2 launch autonomous_rov run_all.launch.py
```

#### Localization with OpenCV Tools

Launch the localization node to detect the agent and goal positions:

```bash
ros2 run opencv_tools img_publisher
ros2 run opencv_tools robot_goal_pos_extimation
```
#### Start the Visdom Server
```bash
python -m visdom.server
```
This will start the server on 'localhost:8097' by default.

#### DDDQN Training and Testing

Launch the training or testing scripts for the DDDQN method:

```bash
ros2 run rov_practical dddqn
ros2 run rov_practical uw
```

### Visualizing with Rviz2

You can visualize the tf published by the `opencv_tools` package in Rviz2:

```bash
rviz2
```

## Running Tests

To run the tests for any package, use the following command:

```bash
colcon test --packages-select <package_name>
```

For example, to test the `autonomous_rov` package:

```bash
colcon test --packages-select autonomous_rov
```

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions, feel free to open an issue or contact me directly at mdeather0056@gmail.com.