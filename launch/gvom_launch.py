#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define the path to the parameters file
    params_file_path = os.path.join(
        get_package_share_directory('gvom'),
        'config',
        'gvom_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='gvom',
            name='gvom_ros2',
            executable='gvom',
            output='screen',
            namespace='gvom',
            parameters=[params_file_path],
            remappings=[
                ('lidar_points', '/warty/lidar_points'),
                ('odom', '/warty/odom'),
            ]
        ),
    ])