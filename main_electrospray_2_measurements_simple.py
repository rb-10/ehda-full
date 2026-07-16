"""
Electrospray Mapping  –  Main entry point (Simplified for Custom Frequencies)
VERSION: 4.0 (Simplified)

This version removes the signal processing and classification steps that break
when using different sampling frequencies. It simply loops through the 
voltages/flows, acquires the raw data, and saves it to the database and .npy files.

Timing per voltage step:
  set_voltage → wait stab_time → acquire (oscilloscope) → save 
              → wait step_time → next voltage
"""

import sys
import time
import warnings
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from mapping.software.electrospray        import ElectrosprayConfig
from mapping.software.hardware            import Hardware
from mapping.software.database            import ElectrosprayDatabase
from mapping.software.camera              import CameraClassifier

warnings.filterwarnings("ignore")

# --- CONFIGURATION OVERRIDES ---
CUSTOM_SAMPLE_RATE = 100000  # Set your desired frequency here
CUSTOM_RECORD_LENGTH = 50000 # Set the number of samples you want to acquire

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

def get_experiment_metadata():
    print("\n" + "="*30)
    print(" NEW ELECTROSPRAY SESSION (SIMPLE)")
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

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────
    config_obj = ElectrosprayConfig("mapping/setup/mapsetup.json")
    config_obj.load_json_config_setup()
    cfg  = config_obj.get_json_setup()
    meas = cfg["typeofmeasurement"]

    stab_time      = float(meas.get("stab_time",              3.0))
    step_time      = float(meas.get("step_time",              5.0))
    flow_stab_time = float(cfg.get("flow_stabilization_time", 3.0))

    # ── Arduino trigger ───────────────────────────────────────────────
    camera = CameraClassifier(
        com_port_idx = cfg.get("arduino_com_port", 0),
        model_path   = None, 
    )
    trigger_fn = camera._trigger

    metadata = get_experiment_metadata()
    SESSION_SOLUTION = metadata["solution"]
    SESSION_HV = metadata["hv_position"]
    SESSION_START = datetime.now() 
    
    # ── Hardware ──────────────────────────────────────────────────────
    hardware = Hardware(cfg)
    
    # Override TiePie settings after hardware initializes it
    hardware.scp.sample_rate = CUSTOM_SAMPLE_RATE
    hardware.scp.record_length = CUSTOM_RECORD_LENGTH
    print(f"[MAIN] TiePie Sample Rate set to {CUSTOM_SAMPLE_RATE} Hz")

    # ── Storage ───────────────────────────────────────────────────────
    db = ElectrosprayDatabase(cfg["save_path"])

    steps         = voltage_steps(meas)
    total_points  = len(steps) * len(meas["flow_rate"])
    time_per_step = stab_time + (0.5 * 5) + step_time
    estimated_min = (total_points * time_per_step +
                     len(meas["flow_rate"]) * flow_stab_time) / 60

    print(f"\n[MAIN] Voltage:      {meas['voltage_start']} → {meas['voltage_stop']} V  "
          f"({len(steps)} steps of {meas['step_size']} V)")
    print(f"[MAIN] Flow rates:   {meas['flow_rate']} µL/min")
    print(f"[MAIN] Timing:       stab={stab_time}s  acquire x 5  "
          f"step={step_time}s  flow_stab={flow_stab_time}s")
    print(f"[MAIN] Total points: {total_points}  "
          f"estimated ≥{estimated_min:.1f} min")
    print(f"[MAIN] Trigger:      {'Arduino ready' if camera.arduino else 'no Arduino — trigger disabled'}")
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
                      end="  \n", flush=True)

                hardware.set_voltage(voltage)
                time.sleep(stab_time)

                for measurement_idx in range(2):
                    if abort or keyboard.is_pressed("q"):
                        print("[MAIN] Q pressed – aborting cleanly during measurement loop")
                        abort = True
                        break
                    
                    current_trigger_fn = trigger_fn if measurement_idx == 1 else None

                    # Use our simple_acquire function instead of acquire_and_process
                    result = simple_acquire(
                        hardware.scp,
                        voltage,
                        flow_rate,
                        hardware.actual_voltage(),
                        hardware.actual_current(),
                        trigger_fn = current_trigger_fn
                    )
                    
                    result["solution_name"] = SESSION_SOLUTION
                    result["hv_position"] = SESSION_HV
                    
                    # 5. Save to database and .npy
                    db.save(result)

                    print(f"         Meas {measurement_idx+1}/5: I={result['mean_na']:.3f} nA (Saved without ML)")

                if abort:
                    break
                    
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
        print(f"Save video with this name: \n{db.finalize_session(SESSION_SOLUTION, SESSION_START).rsplit('.', 1)[0]}")
        db.close()

    print(f"\n[MAIN] Done.  Results saved to: {cfg['save_path']}")
    plt.ioff()
    sys.exit(0)
