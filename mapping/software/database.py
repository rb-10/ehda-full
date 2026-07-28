import os
import sqlite3
import numpy as np
import pandas as pd

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
    qty_max             INTEGER,
    pct_max             REAL,
    sample_rate         REAL,
    n_samples           INTEGER,
    rf_spray_mode       TEXT,
    xgb_spray_mode      TEXT,
    image_classification  TEXT,
    manual_classification TEXT,
    video_file          TEXT,
    raw_data_file       TEXT
);
"""

# Now 19 columns, 19 placeholders
_INSERT = """
INSERT INTO measurements (
    timestamp, solution_name, hv_position, target_voltage, actual_voltage, actual_current_ps,
    flow_rate, mean_na, deviation_na, qty_max, pct_max, sample_rate, n_samples,
    rf_spray_mode, xgb_spray_mode, image_classification, manual_classification,
    video_file, raw_data_file
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_MIGRATIONS = [
    ("ml_spray_mode",         "rf_spray_mode",         "TEXT", "'N/A'"),
    ("nn_spray_mode",         "xgb_spray_mode",        "TEXT", "'N/A'"),
    ("qty_max",               "qty_max",               "INTEGER", "0"),
    ("pct_max",               "pct_max",               "REAL", "0.0"),
    ("image_classification",  "image_classification",  "TEXT", "'N/A'"),
    ("manual_classification", "manual_classification", "TEXT", "'N/A'"),
    ("sample_rate",           "sample_rate",           "REAL", "100000.0"),
    ("n_samples",             "n_samples",             "INTEGER", "50000"),
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
            float(result.get("mean_na", 0)),
            float(result.get("deviation_na", 0)),
            int(result.get("qty_max", 0)),
            float(result.get("pct_max", 0)),
            float(result.get("sample_rate", 100000.0)),
            int(result.get("n_samples", 50000)),
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
    
    def load_training_dataframe(self) -> pd.DataFrame:
        """
        Interactively selects solutions and returns a DataFrame for ML training.
        """
        # 1. Get unique solutions
        query_solutions = "SELECT DISTINCT solution_name FROM measurements WHERE solution_name IS NOT NULL"
        solutions = [row['solution_name'] for row in self._conn.execute(query_solutions).fetchall()]
        
        if not solutions:
            print("[DB] No data found in database.")
            return pd.DataFrame()

        # 2. Display and Ask
        print("\n--- Available Solutions in Database ---")
        for i, sol in enumerate(solutions):
            print(f"[{i}] {sol}")
        
        choice = input("\nEnter indexes to use (e.g., '0, 2'), or press Enter for ALL: ").strip()
        
        # 3. Build Query based on selection
        base_query = "SELECT * FROM measurements"
        
        if choice:
            try:
                # Convert string "0, 5, 6" to list of integers
                indexes = [int(x.strip()) for x in choice.split(',')]
                selected_solutions = [solutions[i] for i in indexes]
                
                # Format for SQL IN clause: ('Sol1', 'Sol2')
                placeholders = ', '.join(['?'] * len(selected_solutions))
                query = f"{base_query} WHERE solution_name IN ({placeholders})"
                
                print(f"[DB] Loading samples for: {selected_solutions}")
                df = pd.read_sql_query(query, self._conn, params=selected_solutions)
            except (ValueError, IndexError):
                print("[Error] Invalid input. Loading ALL data instead.")
                df = pd.read_sql_query(base_query, self._conn)
        else:
            print("[DB] Loading all available samples.")
            df = pd.read_sql_query(base_query, self._conn)

        print(f"[DB] Loaded {len(df)} samples.")
        return df
    def get_previous_step_mean_na(self, current_id, target_voltage: float, flow_rate: float, max_steps_back: int = 50):
        """
        `current_id` is the id of the most recently inserted row (the current
        sample itself has not been inserted yet, so there is no id for it).
        Looks backward *including* current_id for the most recent step whose
        target_voltage differs from `target_voltage` (the upcoming sample's
        voltage), capped at `max_steps_back` step-transitions.
        """
        cursor = self._conn.execute(
            """
            SELECT id, target_voltage, flow_rate, mean_na FROM measurements
            WHERE id <= ?
            ORDER BY id DESC
            """,
            (current_id,)
        )
    
        current_step_voltage = target_voltage  # the voltage the NEW sample will have
        step_count = 0
        prev_voltage = None
        prev_flow_rate = None
        vals = []
    
        for r in cursor:
            v = r["target_voltage"]
    
            if v != current_step_voltage:
                step_count += 1
                current_step_voltage = v
    
                if step_count > max_steps_back:
                    break
                
                if step_count == 1:
                    prev_voltage = v
                    prev_flow_rate = r["flow_rate"]
    
            if step_count == 1:
                if r["mean_na"] is not None:
                    vals.append(r["mean_na"])
            elif step_count >= 2:
                break
            
        if prev_voltage is None or prev_flow_rate != flow_rate or not vals:
            return None
    
        return float(np.mean(vals))

    def get_previous_step_classification(self, current_id, target_voltage: float, flow_rate: float):
        """
        Returns the rf_spray_mode of the most-recent measurement that belongs
        to a *different* voltage step (different target_voltage, same flow_rate),
        looking at rows with id <= current_id.

        Returns None if no such previous step exists — this is the case when
        the current measurement is the first step of the flow-rate sweep, so
        callers can distinguish "no previous step" from a non-multi_jet label.
        """
        row = self._conn.execute(
            """
            SELECT id, rf_spray_mode FROM measurements
            WHERE id <= ? AND target_voltage != ? AND flow_rate = ?
            ORDER BY id DESC LIMIT 1
            """,
            (current_id, target_voltage, flow_rate)
        ).fetchone()
        
        if row is None:
            return None  # no earlier step with a different voltage → first step of sweep
        elif abs(int(row["id"]) - current_id) > 10:
            return None
        else:
            return row["rf_spray_mode"]

    def get_last_id(self):
        """Returns the highest id currently in the table, or None if empty."""
        row = self._conn.execute("SELECT MAX(id) AS max_id FROM measurements").fetchone()
        return row["max_id"] if row and row["max_id"] is not None else None