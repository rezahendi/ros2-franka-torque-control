#!/usr/bin/env python3
"""Sim-to-real robustness study, run without ROS 2.

Uses exactly the same core code as the ROS 2 nodes (plant, controllers, trajectory), but steps
everything in one deterministic loop: physics at 1 kHz, controller at 500 Hz (zero-order hold).
Writes results/results.csv, results/results.md and results/tracking_error.png.

Usage:
    export MENAGERIE_PATH=~/mujoco_menagerie
    python3 scripts/offline_experiment.py [--duration 15]
"""
import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "franka_torque_control"))
from franka_torque_control.core import (DynamicsModel, MinJerkWaypoints, MujocoPlant,  # noqa: E402
                                        PlantConfig, make_controller)
from franka_torque_control.core.metrics import max_error_deg, rms_error_deg  # noqa: E402

NOMINAL = PlantConfig()
SCENARIOS = {
    "nominal": NOMINAL,
    "payload 1.5 kg": replace(NOMINAL, payload_kg=1.5),
    "link masses +15%": replace(NOMINAL, link_mass_scale=1.15),
    "joint friction": replace(NOMINAL, friction_torque=0.5, extra_damping=2.0),
    "actuation delay 3 ms": replace(NOMINAL, actuation_delay_s=0.003),
    "sensor noise": replace(NOMINAL, sensor_noise_pos=5e-4, sensor_noise_vel=5e-3),
    "all combined": replace(NOMINAL, payload_kg=1.5, link_mass_scale=1.15, friction_torque=0.5,
                            extra_damping=2.0, actuation_delay_s=0.003,
                            sensor_noise_pos=5e-4, sensor_noise_vel=5e-3),
}
CONTROLLERS = ["pd_gravity", "computed_torque"]


def run(controller: str, cfg: PlantConfig, duration: float, control_rate: float = 500.0):
    plant = MujocoPlant(cfg)
    ctrl = make_controller(controller, DynamicsModel())
    traj = MinJerkWaypoints.around_home()
    every = max(1, int(round(1.0 / (control_rate * cfg.timestep))))
    tau = np.zeros(7)
    q_des_log, q_log = [], []
    for i in range(int(duration / cfg.timestep)):
        if i % every == 0:
            q, qd = plant.measure()
            tau = ctrl(q, qd, *traj.sample(plant.time))
        plant.step(tau)
        q_des_log.append(traj.sample(plant.time)[0])
        q_log.append(plant.true_state()[0])
    return rms_error_deg(q_des_log, q_log), max_error_deg(q_des_log, q_log)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=15.0, help="simulated seconds per run")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario, cfg in SCENARIOS.items():
        for controller in CONTROLLERS:
            rms, worst = run(controller, cfg, args.duration)
            rows.append({"scenario": scenario, "controller": controller,
                         "rms_deg": float(np.mean(rms)), "max_deg": worst})
            print(f"{scenario:22s} {controller:16s} RMS {np.mean(rms):7.3f} deg   max {worst:7.3f} deg")

    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    table = {(r["scenario"], r["controller"]): r for r in rows}
    lines = ["| Scenario | PD + gravity: RMS / max [deg] | Computed torque: RMS / max [deg] |",
             "|---|---|---|"]
    for scenario in SCENARIOS:
        cells = [f"{table[(scenario, c)]['rms_deg']:.3f} / {table[(scenario, c)]['max_deg']:.3f}"
                 for c in CONTROLLERS]
        lines.append(f"| {scenario} | {cells[0]} | {cells[1]} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(SCENARIOS))
        fig, ax = plt.subplots(figsize=(9, 4))
        for k, controller in enumerate(CONTROLLERS):
            vals = [table[(s, controller)]["rms_deg"] for s in SCENARIOS]
            ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38, label=controller)
        ax.set_xticks(x)
        ax.set_xticklabels(list(SCENARIOS), rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("mean RMS joint error [deg] (log)")
        ax.set_title("Tracking error under sim-to-real effects")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "tracking_error.png", dpi=150)
    except ImportError:
        print("matplotlib not installed: skipping the plot")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
