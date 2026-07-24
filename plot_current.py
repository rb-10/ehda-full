import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import keyboard

# Multiplier for raw data values (e.g., to convert from A to nA, use 1e9)
MULTIPLIER = 0.002

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mapping.software.database import ElectrosprayDatabase

def main():
    BASE = project_root / "data"
    RAW_DIR = BASE / "raw_waveforms"
    
    print(f"Loading database from: {BASE}")
    db = ElectrosprayDatabase(str(BASE))
    
    # db.load_training_dataframe() interactively asks for solutions
    df = db.load_training_dataframe()
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
        
    print(f"\nLoaded {len(df)} samples.")
    print("Controls:")
    print("  'n' - Next sample")
    print("  'p' - Previous sample")
    print("  'q' or 'esc' - Quit\n")
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Use a list to store the current index so it can be updated inside the loop
    current_idx = [0]
    
    def update_plot():
        idx = current_idx[0]
        ax.clear()
        row = df.iloc[idx]
        file_path = RAW_DIR / str(row["raw_data_file"])
        
        sol_name = row.get("solution_name", "Unknown")
        voltage = row.get("actual_voltage", "Unknown")
        flow = row.get("flow_rate", "Unknown")
        
        # Add solution_name, actual_voltage, and flow_rate to the legend
        label_str = f"solution_name: {sol_name}\nactual_voltage: {voltage}\nflow_rate: {flow}"
        
        if file_path.exists():
            try:
                raw_data = np.load(file_path) * MULTIPLIER
                ax.plot(raw_data, label=label_str, color='#2e62d4', alpha=0.8, linewidth=0.5)
                ax.legend(loc="upper right", fontsize=10)
                ax.set_title(f"Sample {idx + 1} / {len(df)} - ID: {row.get('id', '?')}")
                ax.set_xlabel("Sample index")
                ax.set_ylabel("Voltage (V)")
                # Force the Y-axis to match the TiePie software's +/- 4V view!
                ax.set_ylim(-4.0, 4.0)
                ax.grid(True, linestyle="--", alpha=0.5)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading file: {e}", ha="center", va="center", color="red")
                ax.set_title(f"Sample {idx + 1} / {len(df)} - ID: {row.get('id', '?')}")
        else:
            ax.text(0.5, 0.5, "File not found", ha="center", va="center", color="red")
            ax.set_title(f"Sample {idx + 1} / {len(df)} - ID: {row.get('id', '?')}")
            
        fig.canvas.draw_idle()

    update_plot()
    
    # Event loop
    while plt.fignum_exists(fig.number):
        try:
            if keyboard.is_pressed('n'):
                if current_idx[0] < len(df) - 1:
                    current_idx[0] += 1
                    update_plot()
                # Debounce: wait until 'n' is released
                while keyboard.is_pressed('n'):
                    plt.pause(0.01)
                    
            elif keyboard.is_pressed('p'):
                if current_idx[0] > 0:
                    current_idx[0] -= 1
                    update_plot()
                # Debounce: wait until 'p' is released
                while keyboard.is_pressed('p'):
                    plt.pause(0.01)
                    
            elif keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                break
                
            plt.pause(0.05)
            
        except Exception:
            # Catch exceptions that might occur if the window is closed
            break

    plt.ioff()
    plt.close('all')

if __name__ == "__main__":
    main()
