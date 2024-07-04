# ROS2 Workspace for Autonomous ROV

This repository contains ROS2 packages for the development and deployment of an autonomous remotely operated vehicle (ROV). The workspace is organized into several packages, including `autonomous_rov`, `opencv_tools`, and `rov_practical`.

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

## Prerequisites

Make sure you have the following installed:

- [ROS2 Foxy Fitzroy](https://docs.ros.org/en/foxy/Installation.html)
- Python 3.8 or later
- [Colcon](https://colcon.readthedocs.io/en/released/)

## Installation

### Clone the Repository

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository/ros2_ws
```

### Install Dependencies

Use the provided `requirements.txt` file to install necessary Python packages:

```bash
pip install -r requirements.txt
```

### Build the Workspace

```bash
source /opt/ros/foxy/setup.bash
colcon build
```

## Usage

### Source the Workspace

Before running any ROS2 nodes, source the workspace:

```bash
source install/setup.bash
```

### Running the Nodes

Navigate to the `launch` directory of the `autonomous_rov` package to launch the ROV:

```bash
ros2 launch autonomous_rov some_launch_file.launch.py
```

### Running Tests

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
```

### Packaging Instructions

To ensure all necessary packages are installed, include a `requirements.txt` file in the root of your repository:

```plaintext
# requirements.txt
numpy
opencv-python
ros2
...
```