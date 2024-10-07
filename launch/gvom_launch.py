#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define the path to the parameters file
    config_file = os.path.join(
        get_package_share_directory('gvom'),
        'config',
        'gvom_params.yaml'
    )

    # Simulated time
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # gvom Node
    gvom_node = Node(
        package='gvom',
        name='gvom_ros2',
        executable='gvom',
        output='screen',
        namespace='gvom',
        parameters=[config_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('lidar_points', '/lester/lidar_points'),
            ('radar_points', '/radar_pts'),
        ]
    )

    ld = LaunchDescription()
    # Set environment variables
    # Add nodes
    ld.add_action(gvom_node)

    return ld