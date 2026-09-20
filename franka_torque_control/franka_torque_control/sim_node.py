"""ROS 2 node that runs the MuJoCo simulation of the Franka arm.

It behaves like the robot would: it publishes measured joint states and applies the joint torques
it receives. Every sim-to-real effect (payload, friction, actuation delay, sensor noise) is set
here through parameters, so the controller never knows about them.

Subscribes: joint_torque_cmd (std_msgs/Float64MultiArray, 7 values, Nm)
Publishes:  joint_states     (sensor_msgs/JointState)
"""
import contextlib

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .core import MujocoPlant, PlantConfig
from .core.model import ARM_JOINTS


class MujocoSimNode(Node):

    def __init__(self):
        super().__init__("mujoco_sim")

        def param(name, default):
            return self.declare_parameter(name, default).value

        self.sim_rate = param("sim_rate_hz", 1000.0)
        self.publish_rate = param("publish_rate_hz", 500.0)
        cfg = PlantConfig(
            timestep=1.0 / self.sim_rate,
            payload_kg=param("payload_kg", 0.0),
            link_mass_scale=param("link_mass_scale", 1.0),
            extra_damping=param("extra_damping", 0.0),
            friction_torque=param("friction_torque", 0.0),
            actuation_delay_s=param("actuation_delay_s", 0.0),
            sensor_noise_pos=param("sensor_noise_pos", 0.0),
            sensor_noise_vel=param("sensor_noise_vel", 0.0),
            seed=param("seed", 0),
        )
        self.plant = MujocoPlant(cfg, param("menagerie_path", ""))
        self.steps_per_tick = max(1, int(round(self.sim_rate / self.publish_rate)))
        self.tau_cmd = np.zeros(7)

        self.state_pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_subscription(Float64MultiArray, "joint_torque_cmd", self.on_torque_cmd, 10)
        self.create_timer(1.0 / self.publish_rate, self.tick)

        self.viewer = None
        if param("use_viewer", True):
            try:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.plant.model, self.plant.data)
            except Exception as exc:                      # no display, no GL, ...
                self.get_logger().warn(f"Viewer disabled ({exc})")
        self._viewer_divider = max(1, int(round(self.publish_rate / 60.0)))
        self._ticks = 0
        self._start_wall = self.get_clock().now()
        self.get_logger().info(
            f"MuJoCo Franka arm: physics at {self.sim_rate:.0f} Hz, "
            f"publishing joint states at {self.publish_rate:.0f} Hz")

    def on_torque_cmd(self, msg: Float64MultiArray):
        if len(msg.data) != 7:
            self.get_logger().warn(f"Ignoring torque command of length {len(msg.data)} (expected 7)",
                                   throttle_duration_sec=2.0)
            return
        self.tau_cmd = np.asarray(msg.data, dtype=float)

    def tick(self):
        lock = self.viewer.lock() if self.viewer is not None else contextlib.nullcontext()
        with lock:
            for _ in range(self.steps_per_tick):
                self.plant.step(self.tau_cmd)

        q, qd = self.plant.measure()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINTS)
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        msg.effort = self.plant.data.ctrl[self.plant.ctrl_idx].tolist()
        self.state_pub.publish(msg)

        self._ticks += 1
        if self.viewer is not None and self._ticks % self._viewer_divider == 0:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.viewer = None
        if self._ticks % int(self.publish_rate * 5) == 0:
            wall = (self.get_clock().now() - self._start_wall).nanoseconds * 1e-9
            self.get_logger().info(f"real-time factor {self.plant.time / max(wall, 1e-9):.2f}")

    def destroy_node(self):
        if self.viewer is not None:
            self.viewer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
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
