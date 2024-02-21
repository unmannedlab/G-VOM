import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define the path to the parameters file
    params_file_path = os.path.join(
        get_package_share_directory('your_package_name'),
        'config',
        'gvom_params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='gvom',
            executable='gvom_ros2.py',
            name='gvom_ros2',
            output='screen',
            parameters=[params_file_path],
            remappings=[
                ('~cloud', '/warty/lidar_points'),
                ('~odom', '/warty/odom'),
            ]
        ),
    ])