import sys
import time
import keyboard
import threading
from mapping.software.electrospray import ElectrosprayConfig
from mapping.software.hardware import Hardware
from mapping.software.camera import CameraClassifier

# --- Hardware Overrides ---
class DemoHardware(Hardware):
    def __init__(self, cfg: dict):
        # Initializing hardware using the provided configuration
        if cfg is None:
            print("[ERROR] Configuration not found!")
            sys.exit(1)
            
        meas = cfg.get("typeofmeasurement", {})
        self._slope = meas.get("slope", 100)
        self._v_start = meas.get("voltage_start", 0)
        
        # Initialize FUG power supply and Syringe pump
        self._init_fug(cfg["fug_com_port"])
        self._init_pump(cfg["pump_com_port"], cfg["diameter syringe"])
        
        # Oscilloscope is removed for this independent demo
        self.scp = None 
        print("[DEMO] Hardware initialized (Pump Reset Logic Active)")

    def _init_scope(self):
        # Overridden to prevent the script from looking for a TiePie device
        pass 

    def update_flow_sequence(self, flow_rate_ul_min: str):
        """
        Forces the pump to stop, updates the rate, and restarts.
        This ensures the pump registers the new flow rate.
        """
        print(f"  -> Stopping pump...")
        self.stop_flow_rate()
        time.sleep(0.5) # Brief pause to allow the pump controller to reset
        
        print(f"  -> Setting new rate: {flow_rate_ul_min} uL/min")
        # Base Hardware method sets the rate and calls start_pumping()
        self.set_flow_rate(flow_rate_ul_min) 
        print(f"  -> Pump restarted.")

# --- Preset Configuration ---
# Modify these values to match your experiment needs
PRESETS = {
    "1": {"name": "Dripping",    "voltage": 1000, "flow": "10.0"},
    "2": {"name": "Micro Dripping", "voltage": 4000, "flow": "5.0"},
    "3": {"name": "Intermitent Jet", "voltage": 5300, "flow": "5.0"},
    "4": {"name": "Cone Jet",   "voltage": 5850, "flow": "5.0"},
    "5": {"name": "Multi Jet",   "voltage": 9500, "flow": "15.0"},
    "6": {"name": "Unstable",   "voltage": 7000, "flow": "30.0"}
}

def input_listener(hw, cam):
    print("\n" + "="*40)
    print("  ELECTROSPRAY PRESET CONTROL ")
    print("="*40)
    for key, val in PRESETS.items():
        print(f" [{key}] {val['name']}: {val['voltage']}V @ {val['flow']} uL/min")
    print(" [T] Trigger Camera | [Q] Exit Demo")
    print("-" * 40)
    
    while True:
        choice = input("\nSelect Preset (1-6) or Command: ").strip().lower()
        
        if choice in PRESETS:
            p = PRESETS[choice]
            print(f"[PRESET {choice}] Applying {p['name']}...")
            
            # 1. Update Voltage on the FUG supply
            
            
            # 2. Update Flow with the required Stop/Start sequence
            hw.update_flow_sequence(p['flow'])
            hw.set_voltage(p['voltage']) 
            print(f"  -> System Ready: {p['voltage']}V | {p['flow']} uL/min")
            
        elif choice == 't':
            cam._trigger()
            print("[TRIGGER] Camera triggered via Terminal.")
            
        elif choice == 'q':
            print("[EXIT] Shutting down...")
            break
        else:
            print("[ERROR] Invalid selection. Choose 1-4, T, or Q.")

if __name__ == "__main__":
    # 1. Load Setup Configuration
    config_obj = ElectrosprayConfig("mapping/setup/mapsetup.json")
    config_obj.load_json_config_setup()
    cfg = config_obj.get_json_setup()

    if cfg is None:
        print("[FATAL] Could not load mapsetup.json. Verify the file path.")
        sys.exit(1)

    # 2. Initialize Hardware and Camera components
    hw = DemoHardware(cfg)
    camera = CameraClassifier(
        com_port_idx = cfg.get("arduino_com_port", 0),
        model_path = None
    )

    # 4. Main Control Loop
    try:
        input_listener(hw, camera)
    except KeyboardInterrupt:
        pass
    finally:
        # Safety shutdown: zeroing voltage and stopping the pump
        hw.shutdown()
        print("[MAIN] Demo finished. Hardware safely shut down.")