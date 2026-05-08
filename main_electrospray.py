"""
Electrospray Mapping  –  Main entry point
VERSION: 4.0

Timing per voltage step:
  set_voltage → wait stab_time → acquire (oscilloscope, ~0.5s min)
              → classify →  save → wait step_time → next voltage

Classification: Random Forest + XGBoost via EHDA flat classifier.
  rf_classification   → RF prediction   e.g. "cone_jet  (87%)"
  xgb_classification  → XGB prediction  e.g. "cone_jet  (91%)"

Models and scalers are loaded from paths defined in mapsetup.json:
  "model_dir"  → folder containing random_forest.pkl, xgboost.pkl,
                  label_encoder.pkl, class_names.pkl, feature_names.pkl
  "scaler_dir" → folder containing the fitted normalizer scalers

Press  Q  at any time to abort cleanly.

Requirements:
    pip install numpy fastai pandas pywavelets scikit-learn joblib keyboard opencv-python python-libtiepie pyserial seaborn xgboost
"""

#General Imports
import sys
import time
import warnings
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from mapping.software.electrospray        import ElectrosprayConfig, ElectrosprayDataProcessing
from mapping.software.hardware            import Hardware
from mapping.software.acquire_and_process import acquire_and_process
from mapping.software.database            import ElectrosprayDatabase
from mapping.software.camera              import CameraClassifier

warnings.filterwarnings("ignore")

#Current Classification Imports
from current_classification.ehda_classifier    import EHDAClassifier
from current_classification.ehda_normalization import EHDAFeatureNormalizer

# ── Helpers ───────────────────────────────────────────────────────────

def voltage_steps(meas: dict) -> list:
    """Return the ordered list of voltage set-points."""
    start = meas["voltage_start"]
    stop  = meas["voltage_stop"]
    step  = abs(meas["step_size"])
    if start <= stop:
        pts = list(np.arange(start, stop + step, step))
    else:
        pts = list(np.arange(start, stop - step, -step))
    return [float(v) for v in pts]


def load_ml_models(cfg: dict) -> dict:
    """
    Load RF and XGBoost classifiers + normalizer from paths in mapsetup.json.

    Expected mapsetup.json keys:
      "model_dir"  : folder with random_forest.pkl, xgboost.pkl, etc.
      "scaler_dir" : folder with amplitude_scaler.pkl, etc.

    Returns a dict with keys "rf", "xgb", "normalizer",
    or an empty dict if loading fails.
    """

    model_dir  = cfg.get("model_dir",  "current_classification/models/")
    scaler_dir = cfg.get("scaler_dir", "current_classification/scalers/")
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

def get_experiment_metadata():
    print("\n" + "="*30)
    print(" NEW ELECTROSPRAY SESSION ")
    print("="*30)
    
    solution = input("Enter solution name (e.g., Ethanol + 0.1M LiCl): ")
    
    print("\nHigh Voltage Configuration:")
    print("1. HV on Nozzle (Counter-Electrode Grounded)")
    print("2. HV on Counter-Electrode (Nozzle Grounded)")
    choice = input("Select (1 or 2): ")
    hv_pos = "nozzle" if choice == "1" else "counter-electrode"
    
    return {
        "solution": solution,
        "hv_position": hv_pos
    }

def classify_sample(processing, result, ml_models):
    if not ml_models:
        return "N/A", "N/A"

    try:
        from current_classification.ehda_normalization import prepare_inference_sample
        import pandas as pd
        # 1. Calculate remaining ML features inside the class
        processing.extract_advanced_ml_features()

        # 2. Build the full raw feature vector (DB stats + ML stats + metadata)
        all_features = processing.get_db_features_dictionary()
        all_features.update(processing.ml_features)
        all_features.update({
            "actual_voltage": float(result["actual_voltage"]),
            "target_voltage": float(result["target_voltage"]),
            "flow_rate": float(result["flow_rate"]),
            "voltage_error": float(result["actual_voltage"]) - float(result["target_voltage"])
        })

        # 3. Normalization Pipeline
        # prepare_inference_sample returns a 1D array aligned to the normalizer's columns
        x_norm = prepare_inference_sample(all_features, ml_models["normalizer"])

        # 4. Final Alignment to Model
        # Since the normalizer may have more columns than the RF model (e.g., 66 vs 61),
        # we re-wrap and select the exact features the RF model expects.
        all_feature_names = ml_models["normalizer"].get_feature_columns()
        df_full = pd.DataFrame([x_norm], columns=all_feature_names)
        
        # Select the 61 features the RF model expects
        rf_features = ml_models["rf"].feature_names
        x_rf_aligned = df_full[rf_features].values[0]

        # 5. Prediction
        rf_result = "N/A"
        if "rf" in ml_models:
            pred, proba = ml_models["rf"].predict(x_rf_aligned)
            rf_result = f"{pred} ({proba.get(pred, 0.0):.0%})"

        xgb_result = "N/A"
        if "xgb" in ml_models:
            # Re-align for XGB if it uses different features than RF
            xgb_features = ml_models["xgb"].feature_names
            x_xgb_aligned = df_full[xgb_features].values[0]
            pred, proba = ml_models["xgb"].predict(x_xgb_aligned)
            xgb_result = f"{pred} ({proba.get(pred, 0.0):.0%})"

        return rf_result, xgb_result

    except Exception as e:
        print(f"[CLASSIFY] Error: {e}")
        return "error", "error"

# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────
    config_obj = ElectrosprayConfig("mapping/setup/mapsetup.json")
    config_obj.load_json_config_setup()
    cfg  = config_obj.get_json_setup()
    meas = cfg["typeofmeasurement"]

    stab_time      = float(meas.get("stab_time",              3.0))
    step_time      = float(meas.get("step_time",              5.0))
    flow_stab_time = float(cfg.get("flow_stabilization_time", 3.0))

    # ── Signal processing ─────────────────────────────────────────────
    processing = ElectrosprayDataProcessing(1e5)

    # ── ML models ─────────────────────────────────────────────────────
    ml_models = load_ml_models(cfg)
    
    # ── Arduino trigger ───────────────────────────────────────────────
    # CameraClassifier connects the Arduino on init.
    # We only need the _trigger method — no image capture used here.
    camera = CameraClassifier(
        com_port_idx = cfg.get("arduino_com_port", 0),
        model_path   = None,   # image model not used
    )
    # Pass the bound trigger method — called inside acquire_and_process
    # exactly when scp.start() fires. If Arduino is unavailable,
    # camera._trigger is a no-op and everything continues normally.
    trigger_fn = camera._trigger

    metadata = get_experiment_metadata()
    SESSION_SOLUTION = metadata["solution"]
    SESSION_HV = metadata["hv_position"]
    SESSION_START = datetime.now() # Capture start time for the final filename
    # ── Hardware ──────────────────────────────────────────────────────
    hardware = Hardware(cfg)

    # ── Storage ───────────────────────────────────────────────────────
    db = ElectrosprayDatabase(cfg["save_path"])

    steps         = voltage_steps(meas)
    total_points  = len(steps) * len(meas["flow_rate"])
    time_per_step = stab_time + 0.5 + step_time
    estimated_min = (total_points * time_per_step +
                     len(meas["flow_rate"]) * flow_stab_time) / 60

    print(f"\n[MAIN] Voltage:      {meas['voltage_start']} → {meas['voltage_stop']} V  "
          f"({len(steps)} steps of {meas['step_size']} V)")
    print(f"[MAIN] Flow rates:   {meas['flow_rate']} µL/min")
    print(f"[MAIN] Timing:       stab={stab_time}s  acquire≥0.5s  "
          f"step={step_time}s  flow_stab={flow_stab_time}s")
    print(f"[MAIN] Total points: {total_points}  "
          f"estimated ≥{estimated_min:.1f} min")
    print(f"[MAIN] Trigger:      {'Arduino ready' if camera.arduino else 'no Arduino — trigger disabled'}")
    print(f"[MAIN] RF loaded:    {'yes' if 'rf'  in ml_models else 'no'}"
          f"   XGB loaded: {'yes' if 'xgb' in ml_models else 'no'}")
    print("[MAIN] Press  Q  to abort\n")

    abort   = False
    counter = 0



    try:
        for flow_rate in meas["flow_rate"]:
            if abort:
                break

            print(f"\n[MAIN] ── Flow rate: {flow_rate} µL/min ──────────────────")
            hardware.set_flow_rate(str(flow_rate))
            print(f"[MAIN]   Waiting {flow_stab_time}s for flow to stabilise...")
            time.sleep(flow_stab_time)
            hardware.pump_beep()

            for voltage in steps:

                if abort or keyboard.is_pressed("q"):
                    print("[MAIN] Q pressed – aborting cleanly")
                    abort = True
                    break

                counter += 1
                print(f"[MAIN]   [{counter}/{total_points}]  "
                      f"{voltage:.0f} V  |  {flow_rate} µL/min",
                      end="  ", flush=True)

                # 1. Set voltage and wait for stabilisation
                hardware.set_voltage(voltage)
                time.sleep(stab_time)

                # 2. Acquire raw waveform + compute signal statistics
                result = acquire_and_process(
                    hardware.scp,
                    voltage,
                    flow_rate,
                    hardware.actual_voltage(),
                    hardware.actual_current(),
                    processing,
                    trigger_fn = trigger_fn  # ← synchronized trigger
                )
                result["solution_name"] = SESSION_SOLUTION
                result["hv_position"] = SESSION_HV
                # 3. Classify
                rf_result, xgb_result = classify_sample(
                    processing,
                    result,
                    ml_models,
                )
                result["rf_classification"]  = rf_result
                result["xgb_classification"] = xgb_result


                # 5. Save  (rf_classification and xgb_classification
                #           are now part of the result dict)
                db.save(result)

                print(f"RF={rf_result}  XGB={xgb_result}  "
                      f"I={result['mean']:.3f} nA")

                # 6. Wait before next step
                time.sleep(step_time)

            hardware.stop_flow_rate()
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt received")

    except Exception as e:
        print(f"\n[MAIN] Unexpected error: {e}")
        import traceback; traceback.print_exc()

    finally:
        hardware.shutdown()
        print(f"Save video with thhis name: \n{db.finalize_session(SESSION_SOLUTION, SESSION_START).rsplit('.', 1)[0]}")
        db.close()

    print(f"\n[MAIN] Done.  Results saved to: {cfg['save_path']}")
    plt.ioff()
    plt.show()
    sys.exit(0)