"""
post_process.py

Reads the SQLite database from a mapping run and exports it to a JSON
file that matches the old save_data.py format.

Usage:
    python post_process.py <save_path>

Example:
    python post_process.py "C:/Users/HV/Desktop/bruno_work/save_electrospray"
"""

import sys
import json
import sqlite3
import numpy as np
import os


def db_to_json(save_path: str):
    db_path   = os.path.join(save_path, "data.db")
    json_path = os.path.join(save_path, "data.json")

    if not os.path.exists(db_path):
        print(f"[POST] Database not found: {db_path}")
        sys.exit(1)

    con  = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row          # lets us access columns by name
    rows = con.execute("SELECT * FROM measurements ORDER BY id").fetchall()
    con.close()

    output = {}
    for row in rows:
        key  = f"sample {row['id'] - 1}"   # match old "sample 0", "sample 1", ...
        data = dict(row)

        # Load the raw waveform and embed it just like the old format did
        raw_file = data.pop("raw_data_file", "")
        if raw_file and os.path.exists(raw_file):
            data["current"] = np.load(raw_file).tolist()
        else:
            data["current"] = []

        # Rename columns to match old JSON field names
        data["voltage"]          = data.pop("actual_voltage",    None)
        data["current_PS"]       = data.pop("actual_current_ps", None)
        data["spray_mode"]       = data.pop("spray_mode",        None)
        data["ml_spray_mode"]    = data.pop("ml_spray_mode",     None)
        data["nn_spray_mode"]    = data.pop("nn_spray_mode",     None)
        data["image_spray_mode"] = data.pop("image_spray_mode",  None)
        data["mean"]             = data.pop("mean_na",           None)
        data["deviation"]        = data.pop("std_na",            None)
        data["median"]           = data.pop("median_na",         None)
        data["rms"]              = data.pop("rms_na",            None)
        data["variance"]         = data.pop("variance_na",       None)

        # Convert any bytes values to hex string for JSON serialization
        for k, v in data.items():
            if isinstance(v, bytes):
                data[k] = v.hex()

        output[key] = data

    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"[POST] Exported {len(rows)} samples → {json_path}")


if __name__ == "__main__":
    # Set your save path here:
    save_path = "C:/Users/HV/Desktop/bruno_work/save_electrospray"  # <-- Change this path as needed
    db_to_json(save_path)