"""Minimum-jerk (quintic) point-to-point motion through a cyclic list of joint-space waypoints."""
import numpy as np

from .model import HOME

# Waypoints as offsets from the home pose [rad]; all stay well inside the Franka joint limits.
DEFAULT_OFFSETS = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, -0.3, 0.4, 0.4, 0.5, 0.4, 0.6],
    [-0.5, 0.3, -0.4, -0.3, -0.5, -0.3, -0.6],
])


def min_jerk(tau: float):
    """Normalized minimum-jerk profile s(tau) and its first two derivatives w.r.t. tau."""
    tau = min(max(tau, 0.0), 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


class MinJerkWaypoints:
    """Moves waypoint -> waypoint (cyclically) with a minimum-jerk profile, holding at each one."""

    def __init__(self, waypoints, segment_time: float = 2.0, hold_time: float = 0.5,
                 start_delay: float = 1.0):
        self.waypoints = np.asarray(waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 7 or len(self.waypoints) < 2:
            raise ValueError("waypoints must have shape (N >= 2, 7)")
        self.T, self.hold, self.delay = segment_time, hold_time, start_delay

    @classmethod
    def around_home(cls, offsets=DEFAULT_OFFSETS, home=HOME, **kwargs):
        return cls(np.asarray(home) + np.asarray(offsets), **kwargs)

    def sample(self, t: float):
        """Desired (q, qd, qdd) at time t [s]."""
        n = len(self.waypoints)
        if t <= self.delay:
            return self.waypoints[0].copy(), np.zeros(7), np.zeros(7)
        period = self.T + self.hold
        k, t_local = divmod(t - self.delay, period)
        q0 = self.waypoints[int(k) % n]
        q1 = self.waypoints[(int(k) + 1) % n]
        s, ds, dds = min_jerk(t_local / self.T)
        dq = q1 - q0
        return q0 + dq * s, dq * ds / self.T, dq * dds / self.T**2
