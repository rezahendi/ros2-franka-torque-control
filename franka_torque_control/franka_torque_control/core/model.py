"""Load the Franka Panda (no gripper) from MuJoCo Menagerie and set it up for torque control.

The Menagerie model ships with position servos. For torque control we turn every actuator into a
pure motor (force = ctrl) and limit ctrl to the Franka joint torque limits.
"""
import os
from pathlib import Path

import mujoco
import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
LINK_BODIES = [f"link{i}" for i in range(1, 8)]

# Joint torque limits of the Franka arm [Nm].
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# Fallback "home" pose (same values as the "home" keyframe in the Menagerie model).
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def default_model_path() -> str:
    """Path to panda_nohand.xml, taken from $MENAGERIE_PATH or ~/mujoco_menagerie."""
    root = os.environ.get("MENAGERIE_PATH", str(Path.home() / "mujoco_menagerie"))
    return str(Path(root) / "franka_emika_panda" / "panda_nohand.xml")


def resolve_model_path(model_path: str = "") -> str:
    path = model_path or default_model_path()
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Franka model not found at '{path}'. Clone MuJoCo Menagerie and set MENAGERIE_PATH "
            "(see README)."
        )
    return path


def load_torque_model(model_path: str = "", timestep: float = 0.001) -> mujoco.MjModel:
    """Load the arm and convert its position servos into torque motors."""
    model = mujoco.MjModel.from_xml_path(resolve_model_path(model_path))
    model.opt.timestep = timestep
    model.actuator_gainprm[:] = 0.0
    model.actuator_gainprm[:, 0] = 1.0          # force = 1 * ctrl
    model.actuator_biasprm[:] = 0.0             # no built-in position feedback
    ctrl_idx = actuator_indices(model)
    model.actuator_ctrlrange[ctrl_idx] = np.stack([-TORQUE_LIMITS, TORQUE_LIMITS], axis=1)
    model.actuator_forcerange[ctrl_idx] = np.stack([-TORQUE_LIMITS, TORQUE_LIMITS], axis=1)
    return model


def joint_ids(model: mujoco.MjModel) -> list:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
    if min(ids) < 0:
        raise ValueError("Model does not contain joint1..joint7")
    return ids


def arm_indices(model: mujoco.MjModel):
    """Indices of the 7 arm joints in qpos and in qvel (dof) vectors."""
    ids = joint_ids(model)
    qpos_idx = np.array([model.jnt_qposadr[j] for j in ids])
    dof_idx = np.array([model.jnt_dofadr[j] for j in ids])
    return qpos_idx, dof_idx


def actuator_indices(model: mujoco.MjModel) -> np.ndarray:
    """Index of the actuator that drives each arm joint, in joint order."""
    ids = joint_ids(model)
    act_for_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    return np.array([act_for_joint[j] for j in ids])


def home_configuration(model: mujoco.MjModel) -> np.ndarray:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key < 0:
        return HOME.copy()
    qpos_idx, _ = arm_indices(model)
    return model.key_qpos[key][qpos_idx].copy()
