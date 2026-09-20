"""ROS 2 node that publishes the desired joint trajectory.

Moves the arm through a cyclic list of joint-space waypoints with minimum-jerk segments and
holds briefly at each one.

Publishes: joint_reference (trajectory_msgs/JointTrajectoryPoint) with positions, velocities
           and accelerations, so the controller can use a feedforward term.
"""
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

from .core import MinJerkWaypoints
from .core.trajectory import DEFAULT_OFFSETS


class TrajectoryNode(Node):

    def __init__(self):
        super().__init__("trajectory")

        def param(name, default):
            return self.declare_parameter(name, default).value

        rate = param("rate_hz", 500.0)
        flat = param("waypoint_offsets", DEFAULT_OFFSETS[1:].flatten().tolist())
        if len(flat) % 7 != 0 or len(flat) == 0:
            raise ValueError("waypoint_offsets must contain a multiple of 7 values")
        offsets = np.vstack([np.zeros(7), np.asarray(flat, dtype=float).reshape(-1, 7)])
        self.traj = MinJerkWaypoints.around_home(
            offsets=offsets,
            segment_time=param("segment_time", 2.0),
            hold_time=param("hold_time", 0.5),
            start_delay=param("start_delay", 1.0),
        )

        self.pub = self.create_publisher(JointTrajectoryPoint, "joint_reference", 10)
        self.t0 = self.get_clock().now()
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"publishing a {len(self.traj.waypoints)}-waypoint minimum-jerk trajectory at "
            f"{rate:.0f} Hz ({self.traj.T:.1f} s per segment)")

    def tick(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        q, qd, qdd = self.traj.sample(t)
        msg = JointTrajectoryPoint()
        msg.positions = q.tolist()
        msg.velocities = qd.tolist()
        msg.accelerations = qdd.tolist()
        msg.time_from_start = Duration(seconds=t).to_msg()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
