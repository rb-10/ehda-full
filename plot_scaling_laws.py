"""
plot_scaling_laws.py
Plots the Gañán-Calvo et al. (2018) scaling laws using cone-jet mode data.
"""

import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mapping.software.database import ElectrosprayDatabase

# =============================================================================
# CONFIGURATION
# =============================================================================
# Paths
DB_DATA_DIR = "DMF"

# Fluid properties (Adjust these to match your actual solution)
# Example values provided for Ethanol
FLUID_PROPS = {
    'sigma': 0.0364,           # Surface tension [N/m]
    'rho': 943.9,             # Density [kg/m^3]
    'k': 0.00015,                # Electrical conductivity [S/m]
    'epsilon_0': 8.854e-12  # Dielectric constant (vacuum * relative) [F/m]
}

# Unit Conversion Multipliers to SI units (m^3/s, A, m)
FLOW_RATE_TO_M3_S = 1e-9 / 60  # µL/min -> m³/s (1 µL = 1e-9 m³, 1 min = 60 s)
CURRENT_TO_AMPS = 1e-9             # Assuming mean_na in DB is nA

# Plot Settings
PLOT_STYLE = 'dark_background'
FIGURE_SIZE = (10, 6)
# =============================================================================

def clean_label(label):
    """Clean image classification labels."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    label = str(label)
    label = re.sub(r"\(.*?\)", "", label).strip()
    return label if label else None

def resolve_label(row):
    """Resolve the final classification label."""
    INVALID_LABELS = ["N/A", "undefined", "unconclusive", "noise", "", None]
    manual = row.get("manual_classification")
    if pd.notna(manual) and manual not in INVALID_LABELS:
        return manual

    fallback = clean_label(row.get("image_classification"))
    return fallback

def compute_scaling_scales(sigma, rho, k, epsilon_0):
    """Compute characteristic scales according to Gañán-Calvo et al. (2018)."""
    d0 = (sigma * (epsilon_0 ** 2) / (rho * k ** 2)) ** (1/3)
    Q0 = sigma * epsilon_0 / (rho * k)
    I0 = sigma * (epsilon_0 / rho) ** 0.5
    return d0, Q0, I0

def main():
    plt.style.use(PLOT_STYLE)
    
    # 1. Connect and load data
    print(f"Connecting to database at '{DB_DATA_DIR}'...")
    db = ElectrosprayDatabase(DB_DATA_DIR)
    
    df_db = db.load_training_dataframe()
    if df_db is None or df_db.empty:
        print("No data loaded. Exiting.")
        return
        
    # 2. Filtering
    print("Resolving labels and filtering for 'cone_jet' mode...")
    df_db["final_label"] = df_db.apply(resolve_label, axis=1)
    df_cj = df_db[df_db['final_label'] == 'cone_jet'].copy()
    
    if df_cj.empty:
        print("No cone-jet records found. Exiting.")
        return
        
    print(f"Found {len(df_cj)} cone-jet records.")
    
    # 3. Voltage summary
    voltage_counts = df_cj['target_voltage'].value_counts().sort_index()
    
    print("\nAvailable target voltages with cone-jet records:")
    print("-" * 40)
    print(f"{'Target Voltage (V)':<20} | {'Count'}")
    print("-" * 40)
    for v, c in voltage_counts.items():
        print(f"{v:<20.2f} | {c}")
    print("-" * 40)
    
    # 4. Interactive selection
    while True:
        try:
            sel_input = input("\nEnter the target_voltage to plot (or 'q' to quit): ").strip()
            if sel_input.lower() == 'q':
                return
            selected_voltage = float(sel_input)
            if selected_voltage in voltage_counts.index:
                break
            else:
                print(f"Voltage {selected_voltage} is not in the list. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    df_selected = df_cj[df_cj['target_voltage'] == selected_voltage].copy()
    print(f"\nSelected voltage: {selected_voltage} V ({len(df_selected)} records).")
    
    # 5. Scaling-law computation
    d0, Q0, I0 = compute_scaling_scales(**FLUID_PROPS)
    print("\nCharacteristic Scales:")
    print(f"d0 = {d0:.2e} m")
    print(f"Q0 = {Q0:.2e} m^3/s")
    print(f"I0 = {I0:.2e} A")
    
    # Check if necessary columns exist
    if 'flow_rate' not in df_selected.columns or 'mean_na' not in df_selected.columns:
        print("Error: Required columns 'flow_rate' or 'mean_na' not found in database.")
        return
        
    # Convert measurements to SI units
    df_selected['Q_SI'] = df_selected['flow_rate'] * FLOW_RATE_TO_M3_S
    df_selected['I_SI'] = df_selected['mean_na'] * CURRENT_TO_AMPS
    
    # Normalize variables
    df_selected['Q_norm'] = df_selected['Q_SI'] / Q0
    df_selected['I_norm'] = df_selected['I_SI'] / I0
    
    print("\nNormalizing measured variables (Flow rate, Current)...")
    print("Note: Droplet diameter plotted if available, else omitted (typically not in this database schema).")
    
    # 6. Plotting
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot experimental data
    plt.scatter(df_selected['Q_norm'], df_selected['I_norm'], 
                label='Experimental (Cone-Jet)', color='#2ecc71', alpha=0.7)
    
    # Sort for plotting line
    df_plot = df_selected.sort_values('Q_norm')
    
    # Theoretical Gañán-Calvo slope ~ (Q/Q0)^0.5
    if not df_plot.empty:
        q_min, q_max = df_plot['Q_norm'].min(), df_plot['Q_norm'].max()
        # Add slight padding
        q_vals = np.linspace(q_min * 0.9, q_max * 1.1, 100)
        
        # Fit a curve y = A * x^0.5 to show the trend
        A = np.mean(df_plot['I_norm'] / np.sqrt(df_plot['Q_norm']))
        i_theoretical = A * np.sqrt(q_vals)
        
        plt.plot(q_vals, i_theoretical, color='#e74c3c', linestyle='--', 
                 label=f'Trend ($I/I_0 \\propto (Q/Q_0)^{{1/2}}$)')
    
    plt.title(f'Gañán-Calvo Scaling Law (Cone-Jet at {selected_voltage} V)')
    plt.xlabel('Dimensionless Flow Rate ($Q / Q_0$)')
    plt.ylabel('Dimensionless Current ($I / I_0$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
