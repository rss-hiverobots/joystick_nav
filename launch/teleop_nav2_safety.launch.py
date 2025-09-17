# File: joystick_teleop.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. joy_node (Joystick driver)
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
        ),
        # 2. teleop_twist_joy node with param file and remapping
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            output='screen',
            parameters=[
                '/home/rss/Documents/02_ROS2/ros2_humble_ws/src/joystick_nav/config/teleop_switch.yaml'
            ],
            remappings=[
                ('/cmd_vel', '/cmd_vel_joy'),
            ]
        ),
        # 3. Custom costmap_joystick node
        Node(
            package='joystick_nav',
            executable='costmap_joystick',
            name='costmap_joystick',
            output='screen',
        ),
    ])
