from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    sim = Node(
        package="go2_isaac_ros2",
        executable="main.py",
        name="isaac_sim",
        output="screen",
    )

    return LaunchDescription(
        [
            sim,
        ]
    )
