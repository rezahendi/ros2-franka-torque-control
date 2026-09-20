"""The simulated robot ("plant"), with switches for typical sim-to-real effects.

The controller keeps its own nominal copy of the model, so any perturbation set here is a
mismatch between what the controller assumes and how the "real" robot behaves.
"""
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from .model import (LINK_BODIES, TORQUE_LIMITS, actuator_indices, arm_indices,
                    home_configuration, load_torque_model)


@dataclass
class PlantConfig:
    timestep: float = 0.001          # physics step [s] (1 kHz)
    payload_kg: float = 0.0          # extra mass carried by link 7 (unknown payload)
    link_mass_scale: float = 1.0     # scales all link masses and inertias (model error)
    extra_damping: float = 0.0       # viscous joint damping added to every joint [Nms/rad]
    friction_torque: float = 0.0     # Coulomb joint friction [Nm]
    actuation_delay_s: float = 0.0   # delay between torque command and applied torque [s]
    sensor_noise_pos: float = 0.0    # std of joint position noise [rad]
    sensor_noise_vel: float = 0.0    # std of joint velocity noise [rad/s]
    seed: int = 0


class MujocoPlant:
    def __init__(self, cfg: PlantConfig = PlantConfig(), model_path: str = ""):
        self.cfg = cfg
        self.model = load_torque_model(model_path, cfg.timestep)
        self.q_idx, self.v_idx = arm_indices(self.model)
        self.ctrl_idx = actuator_indices(self.model)
        self._apply_perturbations()
        self.data = mujoco.MjData(self.model)
        self.data.qpos[self.q_idx] = home_configuration(self.model)
        mujoco.mj_forward(self.model, self.data)
        n_delay = int(round(cfg.actuation_delay_s / cfg.timestep))
        self._cmd_queue = deque([np.zeros(7) for _ in range(n_delay)])
        self._rng = np.random.default_rng(cfg.seed)

    def _apply_perturbations(self):
        m, cfg = self.model, self.cfg
        bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for name in LINK_BODIES]
        for b in bodies:
            m.body_mass[b] *= cfg.link_mass_scale
            m.body_inertia[b] *= cfg.link_mass_scale
        m.body_mass[bodies[-1]] += cfg.payload_kg   # point mass at the link-7 center of mass
        m.dof_damping[self.v_idx] += cfg.extra_damping
        m.dof_frictionloss[self.v_idx] = cfg.friction_torque
        mujoco.mj_setConst(m, mujoco.MjData(m))

    @property
    def time(self) -> float:
        return self.data.time

    def step(self, tau_cmd) -> None:
        """Apply a torque command (after the configured delay) and advance one physics step."""
        tau = np.clip(np.asarray(tau_cmd, dtype=float), -TORQUE_LIMITS, TORQUE_LIMITS)
        self._cmd_queue.append(tau)
        self.data.ctrl[self.ctrl_idx] = self._cmd_queue.popleft()
        mujoco.mj_step(self.model, self.data)

    def true_state(self):
        return self.data.qpos[self.q_idx].copy(), self.data.qvel[self.v_idx].copy()

    def measure(self):
        """Joint positions and velocities as a sensor would report them (with noise)."""
        q, qd = self.true_state()
        if self.cfg.sensor_noise_pos > 0:
            q = q + self._rng.normal(0.0, self.cfg.sensor_noise_pos, 7)
        if self.cfg.sensor_noise_vel > 0:
            qd = qd + self._rng.normal(0.0, self.cfg.sensor_noise_vel, 7)
        return q, qd
