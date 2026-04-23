import os
import sqlite3
import numpy as np

_CREATE = """
CREATE TABLE IF NOT EXISTS measurements (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT,
    solution_name       TEXT,
    hv_position         TEXT,
    target_voltage      REAL,
    actual_voltage      REAL,
    actual_current_ps   REAL,
    flow_rate           REAL,
    mean_na             REAL,
    deviation_na        REAL,
    median_na           REAL,
    rms_na              REAL,
    variance_na         REAL,
    qty_max             INTEGER,
    pct_max             REAL,
    band_power_v_low    REAL,
    band_power_low      REAL,
    band_power_mid      REAL,
    band_power_high     REAL,
    band_power_v_high   REAL,
    rf_spray_mode       TEXT,
    xgb_spray_mode      TEXT,
    image_classification  TEXT,
    manual_classification TEXT,
    video_file          TEXT,
    raw_data_file       TEXT
);
"""

# Now 25 columns, 25 placeholders
_INSERT = """
INSERT INTO measurements (
    timestamp, solution_name, hv_position, target_voltage, actual_voltage, actual_current_ps,
    flow_rate, mean_na, deviation_na, median_na, rms_na, variance_na,
    qty_max, pct_max, 
    band_power_v_low, band_power_low, band_power_mid, band_power_high, band_power_v_high,
    rf_spray_mode, xgb_spray_mode, image_classification, manual_classification,
    video_file, raw_data_file
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_MIGRATIONS = [
    ("ml_spray_mode",         "rf_spray_mode",         "TEXT", "'N/A'"),
    ("nn_spray_mode",         "xgb_spray_mode",        "TEXT", "'N/A'"),
    ("qty_max",               "qty_max",               "INTEGER", "0"),
    ("pct_max",               "pct_max",               "REAL", "0.0"),
    ("band_power_mid",        "band_power_mid",        "REAL", "0.0"), 
    ("image_classification",  "image_classification",  "TEXT", "'N/A'"),
    ("manual_classification", "manual_classification", "TEXT", "'N/A'"),
]

def _migrate(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(measurements)").fetchall()}
    changed = False
    
    for _, new_col, col_type, default in _MIGRATIONS:
        if new_col not in existing:
            conn.execute(f"ALTER TABLE measurements ADD COLUMN {new_col} {col_type} DEFAULT {default}")
            print(f"[DB] Added column: {new_col}")
            changed = True
            
    if changed:
        conn.commit()
        print("[DB] Schema migration complete")

class ElectrosprayDatabase:
    def __init__(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self._raw_dir = os.path.join(save_path, "raw_waveforms")
        os.makedirs(self._raw_dir, exist_ok=True)

        self.db_path = os.path.join(save_path, "data.db")
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(_CREATE)
        self._conn.commit()
        _migrate(self._conn)
        # Needed to get dict-like rows for easy access
        self._conn.row_factory = sqlite3.Row 
        print(f"[DB] Ready: {self.db_path}")

    def save(self, result: dict):
        ts_str = result['timestamp'].strftime('%Y-%m-%d_%H-%M-%S_%f')
        waveform_filename = f"wf_{ts_str}.npy"

        waveform_path = os.path.join(self._raw_dir, waveform_filename)
        if result.get("datapoints") is not None:
            np.save(waveform_path, result["datapoints"])

        self._conn.execute(_INSERT, (
            result["timestamp"].isoformat(),
            result.get("solution_name", "Unknown"), 
            result.get("hv_position", "Unknown"),
            result.get("target_voltage"),
            result.get("actual_voltage"),
            result.get("actual_current_ps"), 
            result.get("flow_rate"),
            float(result.get("mean", 0)),
            float(result.get("deviation", 0)),
            float(result.get("median", 0)),
            float(result.get("rms", 0)),
            float(result.get("variance", 0)),
            int(result.get("qty_max", 0)),
            float(result.get("pct_max", 0)),
            float(result.get("band_power_v_low", 0)),
            float(result.get("band_power_low", 0)),
            float(result.get("band_power_mid", 0)),
            float(result.get("band_power_high", 0)),
            float(result.get("band_power_v_high", 0)),
            result.get("rf_classification", "N/A"),
            result.get("xgb_classification", "N/A"),
            result.get("image_classification", "N/A"),
            result.get("manual_classification", "N/A"),
            "PENDING",   
            waveform_filename
        ))
        self._conn.commit()

    def finalize_session(self, solution_name: str, session_start_time):
        clean_sol = "".join(c for c in solution_name if c.isalnum() or c in (' ', '_')).strip().replace(" ", "_")
        base_name = f"{session_start_time.strftime('%Y-%m-%d_%H-%M-%S')}_{clean_sol}"
        video_filename = f"{base_name}.mp4"

        self._conn.execute(
            "UPDATE measurements SET video_file = ? WHERE video_file = 'PENDING'",
            (video_filename,)
        )
        self._conn.commit()
        return video_filename

    # --- NEW HELPER METHODS FOR CLASSIFICATION ---
    def get_measurements_by_video(self, video_filename: str):
        """Returns all measurement rows associated with a specific main video file, ordered by time."""
        cursor = self._conn.execute("SELECT * FROM measurements WHERE video_file = ? ORDER BY timestamp ASC", (video_filename,))
        return [dict(row) for row in cursor.fetchall()]

    def update_image_classification(self, row_id: int, classification: str):
        self._conn.execute("UPDATE measurements SET image_classification = ? WHERE id = ?", (classification, row_id))
        self._conn.commit()

    def update_manual_classification(self, row_id: int, classification: str):
        self._conn.execute("UPDATE measurements SET manual_classification = ? WHERE id = ?", (classification, row_id))
        self._conn.commit()

    def close(self):
        self._conn.close()
        print("[DB] Closed")