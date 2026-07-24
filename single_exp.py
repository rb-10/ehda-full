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

from mapping.software.electrospray        import ElectrosprayConfig
from mapping.software.hardware            import Hardware
from mapping.software.database            import ElectrosprayDatabase
from mapping.software.camera              import CameraClassifier

warnings.filterwarnings("ignore")

# ── Reuse helpers from main ────────────────────────────────────────────────────

def simple_acquire(scp, target_voltage: float, flow_rate, actual_voltage: float, actual_current_ps: float, trigger_fn=None) -> dict:
    """
    Acquires data directly without using ElectrosprayDataProcessing.
    This avoids the Butterworth filter crashing on different sampling rates.
    """
    scp.start()

    if trigger_fn is not None:
        try:
            trigger_fn()
        except Exception as e:
            print(f"[ACQ] Trigger error: {e}")

    while not scp.is_data_ready:
        time.sleep(0.01)

    raw = scp.get_data()
    timestamp = datetime.now()
    
    # Channel 2 (raw[1]) with 500 multiplier
    datapoints = np.array(raw[1]) * 500  

    return {
        "datapoints":        datapoints,
        "timestamp":         timestamp,
        "target_voltage":    target_voltage,
        "actual_voltage":    actual_voltage,
        "actual_current_ps": actual_current_ps,
        "flow_rate":         float(flow_rate),
        "mean_na":           float(np.mean(datapoints)),
        "deviation_na":      float(np.std(datapoints)),
        "rf_classification": "N/A",
        "xgb_classification": "N/A",
    }


# ── Manual input prompt ────────────────────────────────────────────────────────

def prompt_run_parameters() -> tuple[float, float, int]:
    """Ask the user for voltage, flow rate, and number of measurements."""
    print("\n" + "=" * 40)
    print("  MANUAL ACQUISITION PARAMETERS")
    print("=" * 40)

    while True:
        try:
            voltage       = float(input("  Target voltage   (V)       : "))
            flow_rate     = float(input("  Flow rate        (µL/min)  : "))
            n_samples     = int(input(  "  Number of measurements     : "))
            if n_samples < 1:
                raise ValueError("Must record at least 1 measurement.")
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

    while True:
        try:
            sample_rate = float(input("\n  Enter Sample rate (Hz) [default 100000]: ") or 100000)
            n_samples = int(input("  Enter Record length (samples) [default 50000]: ") or 50000)
            if sample_rate <= 0 or n_samples < 1:
                raise ValueError("Must be positive")
            break
        except ValueError:
            print("  [!] Invalid input. Please enter valid numbers.")

    return {
        "solution": solution,
        "hv_position": hv_pos,
        "sample_rate": sample_rate,
        "n_samples": n_samples
    }


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
    camera     = CameraClassifier(
        com_port_idx = cfg.get("arduino_com_port", 0),
        model_path   = None,
    )
    trigger_fn = camera._trigger

    # ── Session metadata (asked once) ─────────────────────────────────
    metadata         = get_experiment_metadata()
    SESSION_SOLUTION = metadata["solution"]
    SESSION_HV       = metadata["hv_position"]
    SESSION_SAMPLE_RATE = metadata["sample_rate"]
    SESSION_N_SAMPLES   = metadata["n_samples"]
    SESSION_START    = datetime.now()

    hardware = Hardware(cfg)
    
    # Override TiePie settings after hardware initializes it
    hardware.scp.sample_rate = SESSION_SAMPLE_RATE
    hardware.scp.record_length = SESSION_N_SAMPLES

    db       = ElectrosprayDatabase(cfg["save_path"])

    print("[MANUAL] Starting simple manual acquisition.")
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
                  f"|  Measurements={n_samples}  |  Sample Rate={SESSION_SAMPLE_RATE} Hz  |  Record Length={SESSION_N_SAMPLES}  "
                  f"|  Est. ≥{time_estimate/60:.1f} min")

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

                result = simple_acquire(
                    hardware.scp,
                    voltage,
                    flow_rate,
                    hardware.actual_voltage(),
                    hardware.actual_current(),
                    trigger_fn=trigger_fn,
                )
                result["solution_name"]  = SESSION_SOLUTION
                result["hv_position"]    = SESSION_HV
                result["sample_rate"]    = SESSION_SAMPLE_RATE
                result["n_samples"]      = SESSION_N_SAMPLES

                db.save(result)

                print(f"I={result['mean_na']:.3f} nA (Saved without ML)")

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