import sys
import os
import json
import sqlite3
import numpy as np
from datetime import datetime


TIME_GAP_THRESHOLD = 30   # seconds


def load_rows(save_path: str) -> list:
    db_path = os.path.join(save_path, "data.db")
    if not os.path.exists(db_path):
        print(f"[SPLIT] Database not found: {db_path}")
        sys.exit(1)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM measurements ORDER BY id").fetchall()
    con.close()
    return [dict(row) for row in rows]


def parse_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def split_into_experiments(rows: list) -> list[list]:
    """
    Group rows into experiments.
    A new experiment starts when the time gap between consecutive
    samples exceeds TIME_GAP_THRESHOLD seconds.
    """
    if not rows:
        return []

    experiments = []
    current_exp = [rows[0]]

    for i in range(1, len(rows)):
        prev_ts = parse_timestamp(rows[i - 1].get("timestamp"))
        curr_ts = parse_timestamp(rows[i].get("timestamp"))

        if prev_ts and curr_ts:
            gap = (curr_ts - prev_ts).total_seconds()
            if gap > TIME_GAP_THRESHOLD:
                experiments.append(current_exp)
                current_exp = []

        current_exp.append(rows[i])

    experiments.append(current_exp)   # last experiment
    return experiments


def format_row(row: dict) -> dict:
    """Rename DB columns to match the standard JSON field names."""
    raw_file = row.pop("raw_data_file", "")
    if raw_file and os.path.exists(raw_file):
        row["current"] = np.load(raw_file).tolist()
    else:
        row["current"] = []

    row["voltage"]          = row.pop("actual_voltage",    None)
    row["current_PS"]       = row.pop("actual_current_ps", None)
    row["mean"]             = _safe_float(row.pop("mean_na",    None))
    row["deviation"]        = _safe_float(row.pop("std_na",     None))
    row["median"]           = _safe_float(row.pop("median_na",  None))
    row["rms"]              = _safe_float(row.pop("rms_na",     None))
    row["variance"]         = _safe_float(row.pop("variance_na",None))

    return row


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def save_experiments(experiments: list[list], save_path: str):
    out_dir = os.path.join(save_path, "experiments")
    os.makedirs(out_dir, exist_ok=True)

    for idx, exp_rows in enumerate(experiments):
        output = {}
        for sample_idx, row in enumerate(exp_rows):
            output[f"sample {sample_idx}"] = format_row(row)

        # Summary header so you know what each file contains at a glance
        voltages   = [s.get("voltage")   for s in output.values() if s.get("voltage")]
        flow_rates = [s.get("flow_rate") for s in output.values() if s.get("flow_rate")]
        timestamps = [s.get("timestamp") for s in output.values() if s.get("timestamp")]

        meta = {
            "experiment_index": idx,
            "n_samples":        len(exp_rows),
            "timestamp_start":  min(timestamps) if timestamps else None,
            "timestamp_end":    max(timestamps) if timestamps else None,
            "voltage_min":      min(voltages)   if voltages   else None,
            "voltage_max":      max(voltages)   if voltages   else None,
            "flow_rates":       sorted(set(flow_rates)) if flow_rates else None,
        }

        full_output = {"_meta": meta, **output}

        out_path = os.path.join(out_dir, f"experiment_{idx}.json")
        with open(out_path, "w") as f:
            json.dump(full_output, f, indent=4)

        print(f"[SPLIT] experiment_{idx}.json  →  {len(exp_rows)} samples  "
              f"| V: {meta['voltage_min']:.0f}–{meta['voltage_max']:.0f} V  "
              f"| Q: {meta['flow_rates']} µL/min  "
              f"| {meta['timestamp_start']}  →  {meta['timestamp_end']}")

    print(f"\n[SPLIT] Done. {len(experiments)} experiments saved to: {out_dir}")


if __name__ == "__main__":

    save_path   = "C:/Users/HV/Desktop/bruno_work/save_electrospray"
    rows        = load_rows(save_path)
    experiments = split_into_experiments(rows)

    print(f"[SPLIT] {len(rows)} total samples → {len(experiments)} experiments detected\n")

    # List experiments with summary
    for idx, exp_rows in enumerate(experiments):
        timestamps = [row.get("timestamp") for row in exp_rows if row.get("timestamp")]
        voltages = [row.get("actual_voltage") for row in exp_rows if row.get("actual_voltage") is not None]
        flow_rates = [row.get("flow_rate") for row in exp_rows if row.get("flow_rate") is not None]
        print(f"Experiment {idx}: {len(exp_rows)} samples | "
              f"T: {min(timestamps) if timestamps else None} → {max(timestamps) if timestamps else None} | "
              f"V: {min(voltages) if voltages else None}–{max(voltages) if voltages else None} V | "
              f"Q: {sorted(set(flow_rates)) if flow_rates else None} µL/min")

    print("\nOptions:")
    print("[a] Export all experiments to JSON")
    print("[n] Export only experiment n (e.g., 0, 1, ...)")
    print("[q] Quit without exporting")

    choice = input("Enter your choice ([a]/[n]/[q]): ").strip().lower()

    if choice == "a":
        save_experiments(experiments, save_path)
    elif choice.isdigit() and 0 <= int(choice) < len(experiments):
        exp_idx = int(choice)
        save_experiments([experiments[exp_idx]], save_path)
    elif choice == "q":
        print("[SPLIT] Quit without exporting.")
    else:
        print("[SPLIT] Invalid choice. No experiments exported.")