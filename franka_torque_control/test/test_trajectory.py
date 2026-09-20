import numpy as np
import pytest

from franka_torque_control.core.trajectory import MinJerkWaypoints, min_jerk


def test_min_jerk_starts_and_ends_at_rest():
    for tau in (0.0, 1.0):
        _, ds, dds = min_jerk(tau)
        assert ds == pytest.approx(0.0)
        assert dds == pytest.approx(0.0)
    assert min_jerk(0.0)[0] == pytest.approx(0.0)
    assert min_jerk(1.0)[0] == pytest.approx(1.0)


def test_reaches_each_waypoint_at_rest():
    traj = MinJerkWaypoints.around_home(segment_time=2.0, hold_time=0.5, start_delay=1.0)
    q, qd, qdd = traj.sample(1.0 + 2.0)          # end of the first segment
    assert np.allclose(q, traj.waypoints[1])
    assert np.allclose(qd, 0.0)
    assert np.allclose(qdd, 0.0, atol=1e-9)


def test_reference_is_continuous():
    traj = MinJerkWaypoints.around_home()
    q = np.array([traj.sample(t)[0] for t in np.arange(0.0, 20.0, 1e-3)])
    assert np.max(np.abs(np.diff(q, axis=0))) < 2e-3   # never jumps more than 2 mrad per ms


def test_rejects_invalid_waypoints():
    with pytest.raises(ValueError):
        MinJerkWaypoints(np.zeros((1, 7)))
