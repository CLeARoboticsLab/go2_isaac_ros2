from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_prefix
import os


def generate_launch_description():
    package_name = "go2_isaac_ros2"
    script_path = os.path.join(
        get_package_prefix(package_name), "lib", package_name, "main.py"
    )

    sim = ExecuteProcess(
        cmd=[script_path],
        output="screen",
    )

    return LaunchDescription([sim])
