#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    python_executable = LaunchConfiguration("python_executable")
    script_path = LaunchConfiguration("script_path")

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        "python_executable",
        default_value="python",
        description="Python executable to run OSNet. Use this from an activated virtualenv.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "script_path",
        default_value=os.path.expanduser("~/ros2_ws/src/qwen/qwen/osnet_node.py"),
        description="Path to osnet_node.py. Override if you want to run the build copy.",
    ))

    robots = ["tb3_0", "tb3_1"]

    for ns in robots:
        ld.add_action(ExecuteProcess(
            cmd=[
                python_executable,
                script_path,
                "--ros-args",
                "-r", f"__node:={ns}_osnet_similarity",
                "-p", f"robot_id:={ns}",
            ],
            output="screen",
        ))

    return ld
