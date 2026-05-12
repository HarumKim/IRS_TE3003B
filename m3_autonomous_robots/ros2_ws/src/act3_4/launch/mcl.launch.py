from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='act3_4',
            executable='particle_generator',
            name='mcl_particle_sampler',
            output='screen',
        ),
        Node(
            package='act3_4',
            executable='robot_navigator',
            name='robot_navigator',
            output='screen',
        ),
    ])
