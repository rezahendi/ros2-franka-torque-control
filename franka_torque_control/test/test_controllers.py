"""Closed-loop tests on the MuJoCo model (skipped if MuJoCo Menagerie is not installed)."""
import numpy as np
import pytest

from franka_torque_control.core.model import resolve_model_path

try:
    resolve_model_path()
    HAVE_MODEL = True
except FileNotFoundError:
    HAVE_MODEL = False

pytestmark = pytest.mark.skipif(not HAVE_MODEL, reason="set MENAGERIE_PATH to run these tests")

from franka_torque_control.core import (TORQUE_LIMITS, DynamicsModel,  # noqa: E402
                                        MinJerkWaypoints, MujocoPlant, PlantConfig,
                                        make_controller)
from franka_torque_control.core.metrics import rms_error_deg  # noqa: E402


def closed_loop(kind, cfg=PlantConfig(), duration=2.0, traj=None):
    plant = MujocoPlant(cfg)
    ctrl = make_controller(kind, DynamicsModel())
    q_home = plant.true_state()[0]
    tau = np.zeros(7)
    ref, actual = [], []
    for i in range(int(duration / cfg.timestep)):
        if i % 2 == 0:                                   # 500 Hz controller, 1 kHz physics
            q, qd = plant.measure()
            target = traj.sample(plant.time) if traj else (q_home, np.zeros(7), np.zeros(7))
            tau = ctrl(q, qd, *target)
        plant.step(tau)
        ref.append(traj.sample(plant.time)[0] if traj else q_home)
        actual.append(plant.true_state()[0])
    return np.array(ref), np.array(actual)


@pytest.mark.parametrize("kind", ["pd_gravity", "computed_torque"])
def test_holds_home_pose(kind):
    ref, actual = closed_loop(kind)
    assert np.max(np.abs(actual[-1] - ref[-1])) < 1e-3


def test_computed_torque_tracks_with_exact_model():
    ref, actual = closed_loop("computed_torque", duration=6.0,
                              traj=MinJerkWaypoints.around_home())
    assert np.mean(rms_error_deg(ref, actual)) < 0.05


def test_torque_commands_are_clipped():
    plant = MujocoPlant()
    plant.step(np.full(7, 1e3))
    assert np.all(np.abs(plant.data.ctrl[plant.ctrl_idx]) <= TORQUE_LIMITS + 1e-9)


def test_actuation_delay_holds_back_commands():
    plant = MujocoPlant(PlantConfig(actuation_delay_s=0.003))   # 3 physics steps
    applied = []
    for _ in range(4):
        plant.step(np.ones(7))
        applied.append(plant.data.ctrl[plant.ctrl_idx].copy())
    assert np.allclose(applied[:3], 0.0)
    assert np.allclose(applied[3], 1.0)
