"""Start the simulated Franka arm, the torque controller and the trajectory generator."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution([FindPackageShare("franka_torque_control"), "config",
                                   "params.yaml"])
    controller = LaunchConfiguration("controller")
    use_viewer = LaunchConfiguration("use_viewer")

    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value="computed_torque",
                              description="pd_gravity or computed_torque"),
        DeclareLaunchArgument("use_viewer", default_value="true",
                              description="open the MuJoCo viewer window"),
        Node(package="franka_torque_control", executable="sim_node", name="mujoco_sim",
             parameters=[params, {"use_viewer": use_viewer}], output="screen"),
        Node(package="franka_torque_control", executable="controller_node",
             name="torque_controller", parameters=[params, {"controller": controller}],
             output="screen"),
        Node(package="franka_torque_control", executable="trajectory_node", name="trajectory",
             parameters=[params], output="screen"),
    ])
