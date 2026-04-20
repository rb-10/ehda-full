"""
SQLite storage for electrospray mapping results.

Structure
─────────
  <save_path>/
      data.db            – one row per measurement point (stats + classifications)
      raw_waveforms/     – one .npy file per point  (50 k samples @ 100 kHz)

Post-run analysis with pandas
──────────────────────────────
    import sqlite3, pandas as pd
    con = sqlite3.connect("results/data.db")
    df  = pd.read_sql("SELECT * FROM measurements", con)

Column changes from previous version
──────────────────────────────────────
    spray_mode       → removed  (was Sjaak rule-based)
    ml_spray_mode    → rf_spray_mode
    nn_spray_mode    → xgb_spray_mode
    image_spray_mode → removed
"""

import os
import sqlite3
import numpy as np


_CREATE = """
CREATE TABLE IF NOT EXISTS measurements (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT,
    target_voltage      REAL,
    actual_voltage      REAL,
    actual_current_ps   REAL,
    flow_rate           REAL,
    mean_na             REAL,
    std_na              REAL,
    median_na           REAL,
    rms_na              REAL,
    variance_na         REAL,
    rf_spray_mode       TEXT,
    xgb_spray_mode      TEXT,
    raw_data_file       TEXT
);
"""

_INSERT = """
INSERT INTO measurements (
    timestamp, target_voltage, actual_voltage, actual_current_ps,
    flow_rate, mean_na, std_na, median_na, rms_na, variance_na,
    rf_spray_mode, xgb_spray_mode,
    raw_data_file
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_MIGRATIONS = [
    ("ml_spray_mode",   "rf_spray_mode",  "N/A"),
    ("nn_spray_mode",   "xgb_spray_mode", "N/A"),
]


def _migrate(conn):
    existing = {row[1] for row in
                conn.execute("PRAGMA table_info(measurements)").fetchall()}
    changed = False
    for old_col, new_col, default in _MIGRATIONS:
        if old_col in existing and new_col not in existing:
            conn.execute(
                f"ALTER TABLE measurements RENAME COLUMN {old_col} TO {new_col}"
            )
            print(f"[DB] Migrated column: {old_col} -> {new_col}")
            changed = True
        elif new_col not in existing:
            conn.execute(
                f"ALTER TABLE measurements ADD COLUMN {new_col} TEXT DEFAULT '{default}'"
            )
            print(f"[DB] Added missing column: {new_col}")
            changed = True
    if changed:
        conn.commit()
        print("[DB] Schema migration complete")


class ElectrosprayDatabase:

    def __init__(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self._raw_dir = os.path.join(save_path, "raw_waveforms")
        os.makedirs(self._raw_dir, exist_ok=True)

        db_path    = os.path.join(save_path, "data.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE)
        self._conn.commit()
        _migrate(self._conn)
        print(f"[DB] Ready: {db_path}")

    def save(self, result: dict):
        ts       = result["timestamp"].isoformat()
        raw_file = ""

        if result.get("datapoints") is not None:
            fname    = ts.replace(":", "-").replace(".", "-") + ".npy"
            raw_file = os.path.join(self._raw_dir, fname)
            np.save(raw_file, result["datapoints"])

        self._conn.execute(_INSERT, (
            ts,
            result.get("target_voltage"),
            result.get("actual_voltage"),
            result.get("actual_current_ps"),
            result.get("flow_rate"),
            float(result.get("mean",     0)),
            float(result.get("std",      0)),
            float(result.get("median",   0)),
            float(result.get("rms",      0)),
            float(result.get("variance", 0)),
            result.get("rf_classification",  "N/A"),
            result.get("xgb_classification", "N/A"),
            raw_file,
        ))
        self._conn.commit()

    def close(self):
        self._conn.close()
        print("[DB] Closed")