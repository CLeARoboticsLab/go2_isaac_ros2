# go2_isaac_ros2

![animation](media/animation.gif)

This package allows **low-level** (or joint-level), ROS2 control of a Unitree Go2 quadruped robot being simulated in Isaac Sim. This package is built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab). This package is meant to emulate the low-level control mode of the Go2. Here, the robot is controlled by specifying joint position targets and gains to the topic `/lowcmd`. In low-level control mode, sports mode is [disabled](https://support.unitree.com/home/en/developer/Quick_start), so only low-level readings (like joint angles, joint velocities, IMU, and the raw lidar cloud) are available on the `/lowstate` and `/utlidar/cloud` topics.

| Tested With        | Version   |
|--------------------|----------|
| Ubuntu            | 22.04    |
| ROS2              | Humble   |
| Isaac Sim         | 4.5.0    |

TODO:

|  | Feature                      |
|--------|--------------------------------------|
| ✅ | Joint position targets             |
| ⬜ | Joint velocity targets                |
| ✅ | Joint-level specification of Kp and Kd (proportional and derivative gain) |
| ❌ | Joint torque control   |
| ✅ | Joint state and IMU      |
| ✅ | Head LiDAR  |
| ⬜ | Front camera      |
| ⬜ | Terrain selection |

## Installation

First, ensure [ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) is installed.

It is recommended to install Isaac Lab within a conda / miniconda environment. Create an environment:

```bash
conda create -n go2_isaac_ros2 python=3.10
conda activate go2_isaac_ros2
```

Then install [IsaacLab](https://github.com/isaac-sim/IsaacLab) per the their instructions.

Next, create a ROS2 workspace and clone this repository into the `src` directory:

```bash
mkdir -p ~/go2_isaac_ros2_ws/src
cd ~/go2_isaac_ros2_ws/src
git clone https://github.com/CLeARoboticsLab/go2_isaac_ros2.git
```

You will also need to clone the Unitree ROS2 SDK into the `src` directory:

```bash
cd ~/go2_isaac_ros2_ws/src
git clone https://github.com/unitreerobotics/unitree_ros2.git
```

Then build the workspace:

```bash
cd ~/go2_isaac_ros2_ws
source /opt/ros/humble/setup.bash
colcon build
```

⚠️ You may have to build the workspace multiple times to resolve all dependencies.

⚠️ [This PR](https://github.com/isaac-sim/IsaacLab/pull/1809) is required to use the IMU sensor in Isaac Sim. Ensure this PR has been merged into your clone of Isaac Lab.

## Usage

First, start the simulator:

```bash
source ~/go2_isaac_ros2_ws/install/setup.bash
conda activate go2_isaac_ros2
ros2 launch go2_isaac_ros2 launch_sim.py
```

Controlling the robot within Isaac Sim is the same as with hardware. To control the robot, publish `unitree_go.msg.LowCmd` messages to the `/lowcmd` topic.

Similarly, to receive observations from the robot, subscribe to the `/lowstate` and `/utlidar/cloud` topics.

Note: `/clock` is also published by this package to allow for time synchronization with the simulator. To synchronize ROS2 with the simulator, be sure to set the `use_sim_time: true` parameter when launching all nodes.

## Acknowledgements

This package was inspired by the [go2_omniverse](https://github.com/abizovnuralem/go2_omniverse) package by @abizovnuralem.