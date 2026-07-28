import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Goes up 3 levels to 'main/'
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
from mapping.software.database import ElectrosprayDatabase  # Import your DB class
from mapping.software.electrospray import ElectrosprayDataProcessing
# Current Classification Imports
from current_classification.ehda_classifier    import EHDAClassifier
from current_classification.ehda_normalization import EHDAFeatureNormalizer
from current_classification.classify           import classify_sample
from ehda_normalization import prepare_training_data
from ehda_classifier import train


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_FOLDER  = ""
DEFAULT_MODEL_FOLDER  = "current_classification/models"
DEFAULT_SCALER_FOLDER = "current_classification/scalers"

# Set to True to skip samples that already have both classifications saved.
SKIP_ALREADY_CLASSIFIED = False
DEFAULT_SOLUTION_NAME = '200KHZ_TEST2'
DEFAULT_SAMPLING_FREQ = 100_000
DEFAULT_RECORD_LENGTH = 50_000
MULTIPLIER_NA = 500
CUTOFF_HZ     = 3_000
# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_ml_models():
    """
    Load RF and XGBoost classifiers + normalizer.

    Returns a dict with keys "rf", "xgb", "normalizer",
    or an empty dict if loading fails.
    """
    model_dir  = DEFAULT_MODEL_FOLDER
    scaler_dir = DEFAULT_SCALER_FOLDER
    models     = {}

    try:
        models["rf"] = EHDAClassifier.load(model_dir, model_name="random_forest")
        print(f"[MAIN] Random Forest loaded from {model_dir}")
    except Exception as e:
        print(f"[MAIN] Could not load Random Forest: {e}")

    try:
        models["xgb"] = EHDAClassifier.load(model_dir, model_name="xgboost")
        print(f"[MAIN] XGBoost loaded from {model_dir}")
    except Exception as e:
        print(f"[MAIN] Could not load XGBoost: {e}")

    try:
        models["normalizer"] = EHDAFeatureNormalizer.load(scaler_dir)
        print(f"[MAIN] Normalizer loaded from {scaler_dir}")
    except Exception as e:
        print(f"[MAIN] Could not load normalizer: {e}")

    if not models:
        print("[MAIN] No ML models loaded — rf_classification and "
              "xgb_classification will be N/A")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING STATE RESTORATION
# ─────────────────────────────────────────────────────────────────────────────
def restore_processing_from_record(processing, record, db_instance):
    waveform_filename = record.get("raw_data_file")
    if not waveform_filename:
        raise KeyError("No raw_data_file reference found in database record.")

    waveform_path = os.path.join(db_instance._raw_dir, waveform_filename)
    if not os.path.exists(waveform_path):
        raise FileNotFoundError(f"Waveform file missing: {waveform_path}")

    waveform = np.load(waveform_path)

    # 1. Check if the waveform is empty before proceeding
    if waveform.size == 0:
        raise ValueError(f"Waveform file {waveform_filename} is empty.")

    # 2. FORCE-FEED: Assign the waveform to all common internal names 
    # to ensure extract_advanced_ml_features() finds it.
    # ── Filter Design — mirrors acquire_and_process.py ───────────────────
    sample_rate = float(record.get("sample_rate") or processing.sample_rate)

    if sample_rate < 6000:
        processing.datapoints_filtered = waveform
    else:
        cutoff = CUTOFF_HZ / (0.5 * sample_rate)
        b, a = butter(6, Wn=cutoff, btype="low", analog=False)
        processing.calculate_filter(a, b, waveform)

    # ── Signal processing ─────────────────────────────────────────────
    processing.calculate_statistics(processing.datapoints_filtered)
    processing.calculate_power_spectral_density(processing.datapoints_filtered)
    max_val, qty_max, pct_max = processing.calculate_peaks_signal(waveform)
    # Write fresh values back into the record so classify_sample uses them
    # (Rule 2 reads pct_max from the record dict, not from processing)
    record["pct_max"] = float(pct_max)
    record["qty_max"] = int(qty_max)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RECLASSIFICATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Initialise ─────────────────────────────────────────────────────────
    sol_input = input(f"Enter solution name to reclassify [default {DEFAULT_SOLUTION_NAME}]: ").strip()
    solution_name = sol_input if sol_input else DEFAULT_SOLUTION_NAME

    ml_models = load_ml_models()

    if not ml_models:
        print("[MAIN] Aborting — no models could be loaded.")
        return

    # Use the same path your main app uses
    db = ElectrosprayDatabase("data") 

    # ── 2. Fetch all records ──────────────────────────────────────────────────
    # Using pandas via the existing load_training_dataframe logic to get all rows
    print("[MAIN] Fetching records from database...")
    all_records_df = pd.read_sql_query(f"SELECT * FROM measurements WHERE solution_name='{solution_name}'", db._conn)
    all_records = all_records_df.to_dict(orient="records")
    
    total = len(all_records)
    print(f"[MAIN] {total} samples found.")

    if total == 0:
        print(f"[MAIN] No records found for solution '{solution_name}'. Exiting.")
        db.close()
        return

    # ── 3. Counters ───────────────────────────────────────────────────────────
    n_skipped = n_ok = n_error = 0
    n_changed = 0

    # ── 4. Iterate ────────────────────────────────────────────────────────────
    for record in tqdm(all_records, desc="Reclassifying", unit="sample"):
        record_id = record.get("id")
        old_rf = record.get("rf_spray_mode", "")

        sample_rate = float(record.get("sample_rate") or DEFAULT_SAMPLING_FREQ)
        processing = ElectrosprayDataProcessing(sample_rate)

        # ── 4a. Optional skip ─────────────────────────────────────────────────
        if SKIP_ALREADY_CLASSIFIED:
            # Note: Using your actual DB column names here
            rf_saved  = old_rf
            xgb_saved = record.get("xgb_spray_mode", "")
            
            already_done = (
                rf_saved  not in ("", None, "N/A", "error") and
                xgb_saved not in ("", None, "N/A", "error")
            )
            #if already_done:
                #n_skipped += 1
                #continue

        # ── 4b. Restore waveform from disk ────────────────────────────────────
        try:
            restore_processing_from_record(processing, record, db)
        except Exception as e:
            print(f"\n[MAIN] Skip ID {record_id}: {e}")
            n_error += 1
            continue

        # ── 4c. Classify ──────────────────────────────────────────────────────
        # We pass the record dict which contains 'actual_voltage', etc.
        rf_result, xgb_result = classify_sample(processing, record, ml_models, db)

        if rf_result == "error" or xgb_result == "error":
            n_error += 1
            continue
            
        if rf_result != old_rf:
            n_changed += 1

        # ── 4d. Persist results ───────────────────────────────────────────────
        try:
            # We use manual SQL here because ElectrosprayDatabase doesn't have 
            # a generic update() method yet.
            db._conn.execute("""
                UPDATE measurements 
                SET rf_spray_mode = ?, xgb_spray_mode = ? 
                WHERE id = ?
            """, (rf_result, xgb_result, record_id))
            db._conn.commit()
            n_ok += 1
        except Exception as e:
            print(f"\n[DB UPDATE ERROR] ID {record_id}: {e}")
            n_error += 1

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"[MAIN] Reclassification complete.")
    print(f"  Total   : {total}")
    print(f"  OK      : {n_ok}")
    print(f"  Classification Changed: {n_changed}")
    print(f"  Skipped : {n_skipped}  (already classified)")
    print(f"  Errors  : {n_error}")
    print("─" * 60)
    db.close()


if __name__ == "__main__":
    main()
