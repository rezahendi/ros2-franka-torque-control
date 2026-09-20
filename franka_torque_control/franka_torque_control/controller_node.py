"""ROS 2 node that runs the joint-space torque controller at a fixed rate.

Subscribes: joint_states    (sensor_msgs/JointState)
            joint_reference (trajectory_msgs/JointTrajectoryPoint)
Publishes:  joint_torque_cmd          (std_msgs/Float64MultiArray, 7 values, Nm)
            controller/tracking_error (std_msgs/Float64MultiArray, 7 values, rad)

It also reports the measured loop rate and jitter, which is what you want to look at before
trusting any controller on real hardware.
"""
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

from .core import DynamicsModel, make_controller
from .core.controllers import CT_KD, CT_KP, PD_KD, PD_KP
from .core.metrics import loop_timing
from .core.model import ARM_JOINTS


class TorqueControllerNode(Node):

    def __init__(self):
        super().__init__("torque_controller")

        def param(name, default):
            return self.declare_parameter(name, default).value

        kind = param("controller", "computed_torque")
        self.rate = param("control_rate_hz", 500.0)
        gains = {
            "pd_gravity": (param("pd_kp", PD_KP), param("pd_kd", PD_KD)),
            "computed_torque": (param("ct_kp", CT_KP), param("ct_kd", CT_KD)),
        }
        if kind not in gains:
            raise ValueError(f"controller must be one of {list(gains)}, got '{kind}'")
        kp, kd = gains[kind]
        self.controller = make_controller(kind, DynamicsModel(param("menagerie_path", "")), kp, kd)

        self.state = None
        self.reference = None
        self.create_subscription(JointState, "joint_states", self.on_state, 10)
        self.create_subscription(JointTrajectoryPoint, "joint_reference", self.on_reference, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, "joint_torque_cmd", 10)
        self.error_pub = self.create_publisher(Float64MultiArray, "controller/tracking_error", 10)

        self._stamps = deque(maxlen=int(5 * self.rate))
        self.create_timer(1.0 / self.rate, self.update)
        self.get_logger().info(f"{kind} controller running at {self.rate:.0f} Hz")

    def on_state(self, msg: JointState):
        try:
            order = [msg.name.index(name) for name in ARM_JOINTS] if msg.name else list(range(7))
        except ValueError:
            self.get_logger().warn("joint_states does not contain joint1..joint7",
                                   throttle_duration_sec=2.0)
            return
        if len(msg.position) < 7 or len(msg.velocity) < 7:
            return
        self.state = (np.array([msg.position[i] for i in order]),
                      np.array([msg.velocity[i] for i in order]))

    def on_reference(self, msg: JointTrajectoryPoint):
        def field(values):
            return np.asarray(values, dtype=float) if len(values) == 7 else np.zeros(7)
        if len(msg.positions) != 7:
            return
        self.reference = (np.asarray(msg.positions, dtype=float),
                          field(msg.velocities), field(msg.accelerations))

    def update(self):
        self._stamps.append(time.perf_counter())
        if self.state is None or self.reference is None:
            self.get_logger().info("waiting for joint_states and joint_reference ...",
                                   throttle_duration_sec=2.0)
            return

        q, qd = self.state
        q_des, qd_des, qdd_des = self.reference
        tau = self.controller(q, qd, q_des, qd_des, qdd_des)
        self.torque_pub.publish(Float64MultiArray(data=tau.tolist()))
        self.error_pub.publish(Float64MultiArray(data=(q_des - q).tolist()))

        if len(self._stamps) > 10:
            stats = loop_timing(self._stamps)
            self.get_logger().info(
                f"loop {stats['rate_hz']:.1f} Hz | jitter {stats['jitter_ms']:.3f} ms | "
                f"worst period {stats['max_period_ms']:.2f} ms | "
                f"max |error| {np.degrees(np.max(np.abs(q_des - q))):.3f} deg",
                throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = TorqueControllerNode()
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
