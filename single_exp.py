"""
Electrospray Manual Acquisition
VERSION: 1.0

Instead of sweeping voltage/flow from a config, this script prompts the user
for a fixed voltage, flow rate, and number of samples, then records them to
the same database format as the main mapping script.

Usage: python manual_acquire.py
"""

import sys
import time
import warnings
import keyboard
import numpy as np
from datetime import datetime

from mapping.software.electrospray        import ElectrosprayConfig, ElectrosprayDataProcessing
from mapping.software.hardware            import Hardware
from mapping.software.acquire_and_process import acquire_and_process
from mapping.software.database            import ElectrosprayDatabase
from mapping.software.camera              import CameraClassifier

warnings.filterwarnings("ignore")

from current_classification.ehda_classifier    import EHDAClassifier
from current_classification.ehda_normalization import EHDAFeatureNormalizer

# ── Reuse helpers from main ────────────────────────────────────────────────────

def load_ml_models(cfg: dict) -> dict:
    model_dir  = cfg.get("model_dir",  "current_classification/models/")
    scaler_dir = cfg.get("scaler_dir", "current_classification/scalers/")
    models     = {}

    try:
        models["rf"] = EHDAClassifier.load(model_dir, model_name="random_forest")
        print(f"[MANUAL] Random Forest loaded from {model_dir}")
    except Exception as e:
        print(f"[MANUAL] Could not load Random Forest: {e}")

    try:
        models["xgb"] = EHDAClassifier.load(model_dir, model_name="xgboost")
        print(f"[MANUAL] XGBoost loaded from {model_dir}")
    except Exception as e:
        print(f"[MANUAL] Could not load XGBoost: {e}")

    try:
        models["normalizer"] = EHDAFeatureNormalizer.load(scaler_dir)
        print(f"[MANUAL] Normalizer loaded from {scaler_dir}")
    except Exception as e:
        print(f"[MANUAL] Could not load normalizer: {e}")

    return models


def classify_sample(processing, result, ml_models):
    if not ml_models:
        return "N/A", "N/A"

    try:
        from current_classification.ehda_normalization import prepare_inference_sample
        import pandas as pd

        processing.extract_advanced_ml_features()

        all_features = processing.get_db_features_dictionary()
        all_features.update(processing.ml_features)
        all_features.update({
            "actual_voltage": float(result["actual_voltage"]),
            "target_voltage": float(result["target_voltage"]),
            "flow_rate":      float(result["flow_rate"]),
            "voltage_error":  float(result["actual_voltage"]) - float(result["target_voltage"])
        })

        x_norm = prepare_inference_sample(all_features, ml_models["normalizer"])
        all_feature_names = ml_models["normalizer"].get_feature_columns()
        df_full = pd.DataFrame([x_norm], columns=all_feature_names)

        rf_result = "N/A"
        if "rf" in ml_models:
            rf_features    = ml_models["rf"].feature_names
            x_rf_aligned   = df_full[rf_features].values[0]
            pred, proba    = ml_models["rf"].predict(x_rf_aligned)
            rf_result      = f"{pred} ({proba.get(pred, 0.0):.0%})"

        xgb_result = "N/A"
        if "xgb" in ml_models:
            xgb_features   = ml_models["xgb"].feature_names
            x_xgb_aligned  = df_full[xgb_features].values[0]
            pred, proba    = ml_models["xgb"].predict(x_xgb_aligned)
            xgb_result     = f"{pred} ({proba.get(pred, 0.0):.0%})"

        return rf_result, xgb_result

    except Exception as e:
        print(f"[CLASSIFY] Error: {e}")
        return "error", "error"


# ── Manual input prompt ────────────────────────────────────────────────────────

def prompt_run_parameters() -> tuple[float, float, int]:
    """Ask the user for voltage, flow rate, and sample count."""
    print("\n" + "=" * 40)
    print("  MANUAL ACQUISITION PARAMETERS")
    print("=" * 40)

    while True:
        try:
            voltage   = float(input("  Target voltage   (V)       : "))
            flow_rate = float(input("  Flow rate        (µL/min)  : "))
            n_samples = int(input(  "  Number of samples          : "))
            if n_samples < 1:
                raise ValueError("Must record at least 1 sample.")
            break
        except ValueError as e:
            print(f"  [!] Invalid input — {e}. Please try again.\n")

    return voltage, flow_rate, n_samples


def get_experiment_metadata() -> dict:
    print("\n" + "=" * 40)
    print("  EXPERIMENT METADATA")
    print("=" * 40)

    solution = input("  Solution name (e.g. Ethanol + 0.1M LiCl): ")

    print("\n  High Voltage Configuration:")
    print("  1. HV on Nozzle (Counter-Electrode Grounded)")
    print("  2. HV on Counter-Electrode (Nozzle Grounded)")
    choice = input("  Select (1 or 2): ")
    hv_pos = "nozzle" if choice.strip() == "1" else "counter-electrode"

    return {"solution": solution, "hv_position": hv_pos}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Config (hardware / paths only — sweep settings are ignored) ───
    config_obj = ElectrosprayConfig("mapping/setup/mapsetup.json")
    config_obj.load_json_config_setup()
    cfg  = config_obj.get_json_setup()
    meas = cfg["typeofmeasurement"]

    stab_time      = float(meas.get("stab_time", 3.0))
    step_time      = float(meas.get("step_time", 5.0))
    flow_stab_time = float(cfg.get("flow_stabilization_time", 3.0))

    # ── Shared infrastructure ─────────────────────────────────────────
    processing = ElectrosprayDataProcessing(1e5)
    ml_models  = load_ml_models(cfg)

    camera     = CameraClassifier(
        com_port_idx = cfg.get("arduino_com_port", 0),
        model_path   = None,
    )
    trigger_fn = camera._trigger

    # ── Session metadata (asked once) ─────────────────────────────────
    metadata         = get_experiment_metadata()
    SESSION_SOLUTION = metadata["solution"]
    SESSION_HV       = metadata["hv_position"]
    SESSION_START    = datetime.now()

    hardware = Hardware(cfg)
    db       = ElectrosprayDatabase(cfg["save_path"])

    print(f"\n[MANUAL] RF loaded:  {'yes' if 'rf'  in ml_models else 'no'}"
          f"   XGB loaded: {'yes' if 'xgb' in ml_models else 'no'}")
    print("[MANUAL] Press  Q  at any time to abort\n")

    # ── Acquisition loop — runs until the user quits ──────────────────
    run_number = 0
    abort      = False

    try:
        while not abort:

            # Ask for parameters each run (or Q to quit)
            print("\n" + "-" * 40)
            quit_check = input("Press Enter to configure a new run, or Q to quit: ")
            if quit_check.strip().lower() == "q":
                break

            voltage, flow_rate, n_samples = prompt_run_parameters()

            time_estimate = (stab_time + 0.5 + step_time) * n_samples + flow_stab_time
            print(f"\n[MANUAL] Voltage={voltage} V  |  Flow={flow_rate} µL/min  "
                  f"|  Samples={n_samples}  |  Est. ≥{time_estimate/60:.1f} min")

            # Set flow and let it stabilise (done once per run)
            hardware.set_flow_rate(str(flow_rate))
            print(f"[MANUAL] Waiting {flow_stab_time}s for flow to stabilise...")
            time.sleep(flow_stab_time)
            hardware.pump_beep()

            # Set voltage once — it stays fixed for all samples in this run
            hardware.set_voltage(voltage)
            print(f"[MANUAL] Waiting {stab_time}s for voltage to stabilise...")
            time.sleep(stab_time)

            run_number += 1

            for i in range(1, n_samples + 1):

                if abort or keyboard.is_pressed("q"):
                    print("[MANUAL] Q pressed — aborting cleanly")
                    abort = True
                    break

                print(f"[MANUAL]   Sample [{i}/{n_samples}]  "
                      f"{voltage:.0f} V  |  {flow_rate} µL/min",
                      end="  ", flush=True)

                result = acquire_and_process(
                    hardware.scp,
                    voltage,
                    flow_rate,
                    hardware.actual_voltage(),
                    hardware.actual_current(),
                    processing,
                    trigger_fn=trigger_fn,
                )
                result["solution_name"]  = SESSION_SOLUTION
                result["hv_position"]    = SESSION_HV

                rf_result, xgb_result = classify_sample(processing, result, ml_models)
                result["rf_classification"]  = rf_result
                result["xgb_classification"] = xgb_result

                db.save(result)

                print(f"RF={rf_result}  XGB={xgb_result}  "
                      f"I={result['mean_na']:.3f} nA")

                # Wait between samples (skip after the last one)
                if i < n_samples:
                    time.sleep(step_time)

            hardware.stop_flow_rate()
            time.sleep(0.5)
            hardware.set_voltage(5000)
            print(f"[MANUAL] Run {run_number} complete.")

    except KeyboardInterrupt:
        print("\n[MANUAL] Keyboard interrupt received")

    except Exception as e:
        print(f"\n[MANUAL] Unexpected error: {e}")
        import traceback; traceback.print_exc()

    finally:
        hardware.shutdown()
        final_name = db.finalize_session(SESSION_SOLUTION, SESSION_START).rsplit(".", 1)[0]
        print(f"\nSave video with this name:\n{final_name}")
        db.close()

    print(f"\n[MANUAL] Done. Results saved to: {cfg['save_path']}")
    sys.exit(0)