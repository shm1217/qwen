#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    python_executable = LaunchConfiguration("python_executable")
    script_path = LaunchConfiguration("script_path")
    namespace = LaunchConfiguration("namespace")

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

    ld.add_action(DeclareLaunchArgument(
        "namespace",
        default_value="tb3_0",
        description="Namespace of the single robot to run OSNet for.",
    ))


    ld.add_action(ExecuteProcess(
        cmd=[
            python_executable,
            script_path,
            "--ros-args",
            "-r", ["__node:=", namespace, "_osnet_similarity"],
            "-p", ["robot_id:=", namespace],
        ],
        output="screen",
    ))

    return ld
