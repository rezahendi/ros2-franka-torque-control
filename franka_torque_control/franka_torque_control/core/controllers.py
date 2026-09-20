"""Joint-space torque controllers for the 7-DoF arm.

Both controllers use their own *nominal* copy of the robot model. They never see the payload,
friction or delays configured in the plant, which is what makes the sim-to-real tests meaningful.
"""
import mujoco
import numpy as np

from .model import arm_indices, load_torque_model


def _full_mass_matrix(model, data, out):
    """Dense mass matrix; works with both the new and the old mj_fullM signature."""
    try:
        mujoco.mj_fullM(model, data, out)        # newer MuJoCo: mj_fullM(m, d, dst)
    except TypeError:
        mujoco.mj_fullM(model, out, data.qM)     # older MuJoCo: mj_fullM(m, dst, qM)


class DynamicsModel:
    """Evaluates M(q) and h(q, qd) for the arm, where M qdd + h = tau."""

    def __init__(self, model_path: str = ""):
        self.model = load_torque_model(model_path)
        self.data = mujoco.MjData(self.model)
        self.q_idx, self.v_idx = arm_indices(self.model)
        self._M_full = np.zeros((self.model.nv, self.model.nv))

    def terms(self, q, qd):
        self.data.qpos[self.q_idx] = q
        self.data.qvel[self.v_idx] = qd
        mujoco.mj_forward(self.model, self.data)
        _full_mass_matrix(self.model, self.data, self._M_full)
        M = self._M_full[np.ix_(self.v_idx, self.v_idx)].copy()
        # qfrc_bias = Coriolis/centrifugal + gravity; subtracting qfrc_passive compensates the
        # nominal joint damping of the model.
        h = self.data.qfrc_bias[self.v_idx] - self.data.qfrc_passive[self.v_idx]
        return M, h

    def gravity(self, q):
        _, g = self.terms(q, np.zeros(7))
        return g


class PDGravityController:
    """tau = Kp (q_des - q) + Kd (qd_des - qd) + g(q)"""

    name = "pd_gravity"

    def __init__(self, dynamics: DynamicsModel, kp, kd):
        self.dyn, self.kp, self.kd = dynamics, np.asarray(kp, float), np.asarray(kd, float)

    def __call__(self, q, qd, q_des, qd_des, qdd_des):
        return self.kp * (q_des - q) + self.kd * (qd_des - qd) + self.dyn.gravity(q)


class ComputedTorqueController:
    """tau = M(q) [qdd_des + Kp e + Kd e_dot] + h(q, qd)  (feedback linearization)"""

    name = "computed_torque"

    def __init__(self, dynamics: DynamicsModel, kp, kd):
        self.dyn, self.kp, self.kd = dynamics, np.asarray(kp, float), np.asarray(kd, float)

    def __call__(self, q, qd, q_des, qd_des, qdd_des):
        M, h = self.dyn.terms(q, qd)
        a = qdd_des + self.kp * (q_des - q) + self.kd * (qd_des - qd)
        return M @ a + h


# Default gains. PD values are in Nm/rad; computed-torque gains are closed-loop
# (Kp = wn^2, Kd = 2 wn for wn = 20 rad/s, critically damped).
PD_KP = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
PD_KD = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]
CT_KP = [400.0] * 7
CT_KD = [40.0] * 7


def make_controller(kind: str, dynamics: DynamicsModel, kp=None, kd=None):
    if kind == "pd_gravity":
        return PDGravityController(dynamics, PD_KP if kp is None else kp, PD_KD if kd is None else kd)
    if kind == "computed_torque":
        return ComputedTorqueController(dynamics, CT_KP if kp is None else kp,
                                        CT_KD if kd is None else kd)
    raise ValueError(f"Unknown controller '{kind}' (use 'pd_gravity' or 'computed_torque')")
