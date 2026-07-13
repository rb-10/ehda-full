import sys
import time

from mapping.software.electrospray import ElectrosprayConfig
from mapping.software.camera import CameraClassifier
from mapping.software.FUG_functions import FUG_initialize, FUG_sendcommands
from mapping.software.PUMP_functions import (
    PUMP_initialize, set_pump_direction, set_inner_diameter, 
    set_flowrate, start_pumping, stop_pumping, low_motor_noize
)

# =============================================================================
# SETTINGS
# =============================================================================
FLOW_RATE = 15.0             # flow rate in uL/min
VOLTAGE_INITIAL = 4400.0    # starting voltage in V
VOLTAGE_FINAL = 4600.0      # final voltage in V
# =============================================================================

def main():
    # Load configuration
    config_obj = ElectrosprayConfig("mapping/setup/mapsetup.json")
    config_obj.load_json_config_setup()
    cfg = config_obj.get_json_setup()
    meas = cfg["typeofmeasurement"]

    print("\n[SCRIPT] Initializing hardware (Bypassing Oscilloscope)...")
    
    # Init Pump
    pump_port = cfg.get("pump_com_port")
    syringe_diameter = cfg.get("diameter syringe")
    pump = PUMP_initialize(pump_port)
    if pump is None:
        print("[SCRIPT] FATAL: Cannot connect to pump")
        sys.exit(1)
    set_pump_direction(pump, "INF")
    set_inner_diameter(pump, syringe_diameter)
    low_motor_noize(pump)
    print("[SCRIPT] Pump ready")

    # Init FUG Power Supply
    fug_port = cfg.get("fug_com_port")
    slope = meas.get("slope", 1000)
    fug = FUG_initialize(fug_port)
    if fug is None:
        print("[SCRIPT] FATAL: Cannot connect to FUG power supply")
        sys.exit(1)
    FUG_sendcommands(fug, [
        ">S1B 0", "I 600e-6", ">S0B 0",
        f">S0R {slope}",
        f"U 0", "F1"
    ])
    print("[SCRIPT] FUG Power Supply ready")
    
    # Initialize Camera Trigger (Arduino)
    camera = CameraClassifier(
        com_port_idx = cfg.get("arduino_com_port", 0),
        model_path   = None
    )
    trigger_fn = camera._trigger

    try:
        # 1. Set flow rate
        print(f"[SCRIPT] Setting flow rate to {FLOW_RATE} uL/min...")
        set_flowrate(pump, str(FLOW_RATE), "UM")
        time.sleep(0.5)
        start_pumping(pump)
        time.sleep(5)
        # 2. Set INITIAL voltage
        print(f"[SCRIPT] Setting INITIAL voltage to {VOLTAGE_INITIAL} V...")
        FUG_sendcommands(fug, [f"U {VOLTAGE_INITIAL}"])
        
        # 3. Wait 6 seconds
        print("[SCRIPT] Waiting 6 seconds for stabilization...")
        time.sleep(6)
        
        # 4. Trigger text and camera
        print("\n" + "="*50)
        print(" >>> TRIGGER NOW! START YOUR RECORDING! <<< ")
        print("="*50 + "\n")
        
        if trigger_fn is not None:
            try:
                trigger_fn()
            except Exception as e:
                print(f"[SCRIPT] Camera trigger error: {e}")
                
        # 5. Wait 1 second
        print("[SCRIPT] Waiting 1 second...")
        time.sleep(2)
        
        # 6. Change voltage to FINAL voltage
        print(f"[SCRIPT] Changing voltage to FINAL voltage {VOLTAGE_FINAL} V...")
        FUG_sendcommands(fug, [f"U {VOLTAGE_FINAL}"])
        
        # Wait for the remainder of the 10-second recording
        print("[SCRIPT] Waiting 9 seconds for you to finish your 10s recording...")
        time.sleep(9)
            
        print("[SCRIPT] Sequence complete.")

    except KeyboardInterrupt:
        print("\n[SCRIPT] Keyboard interrupt received")
    except Exception as e:
        print(f"\n[SCRIPT] Unexpected error: {e}")
        import traceback; traceback.print_exc()
        
    finally:
        print("[SCRIPT] Shutting down hardware safely...")
        try:
            FUG_sendcommands(fug, ["U 0"])
        except Exception:
            pass
        try:
            stop_pumping(pump)
        except Exception:
            pass
        print("[SCRIPT] Hardware shut down.")

if __name__ == "__main__":
    main()
