"""Small helpers to quantify tracking quality and control-loop timing."""
import numpy as np


def rms_error_deg(q_des_log, q_log) -> np.ndarray:
    """Per-joint RMS tracking error in degrees."""
    err = np.asarray(q_des_log) - np.asarray(q_log)
    return np.degrees(np.sqrt(np.mean(err**2, axis=0)))


def max_error_deg(q_des_log, q_log) -> float:
    """Largest absolute joint error over the run, in degrees."""
    return float(np.degrees(np.max(np.abs(np.asarray(q_des_log) - np.asarray(q_log)))))


def loop_timing(stamps_s) -> dict:
    """Mean rate, jitter (std of the period) and worst period of a periodic loop."""
    periods = np.diff(np.asarray(stamps_s, dtype=float))
    if len(periods) == 0:
        return {"rate_hz": float("nan"), "jitter_ms": float("nan"), "max_period_ms": float("nan")}
    return {
        "rate_hz": float(1.0 / np.mean(periods)),
        "jitter_ms": float(1e3 * np.std(periods)),
        "max_period_ms": float(1e3 * np.max(periods)),
    }
