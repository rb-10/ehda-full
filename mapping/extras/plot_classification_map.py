import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent  # Goes up 3 levels to 'main/'
sys.path.insert(0, str(project_root))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import alphashape
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path


from mapping.software.database import ElectrosprayDatabase
# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
BASE = Path(r'C:\Users\HV\Desktop\bruno_work\main\data')
DB_PATH = str(BASE) # Assuming database file logic is handled inside ElectrosprayDatabase
SOLUTION = "EWG343"
# The four sources to plot
PLOT_SOURCES = [
    'manual_classification', 
    'image_classification', 
    'xgb_spray_mode', 
    'rf_spray_mode'
]

# Standard color palette
class_palette = {
    'dripping': "#011cfa",
    'intermitent': "#ffa704",
    'cone_jet': "#0a7f02",
    'multi_jet': "#a13195",
    'undefined': "#000000",
    'unclassified': "#00FFDD",
    'unconclusive': "#7d7e7b",
    'EXCLUDE': '#7f7f7f',
}

# ---------------------------------------------------------
# DATA LOADING & CLEANING
# ---------------------------------------------------------
print("Open DB")
db = ElectrosprayDatabase(DB_PATH)
query = f"SELECT flow_rate, actual_voltage as voltage, image_classification, manual_classification, xgb_spray_mode, rf_spray_mode FROM measurements WHERE solution_name = '{SOLUTION}'"

# Load DB into DataFrame
df_raw = pd.read_sql(query, db._conn)
print("Close DB")
db.close()

def clean_label(val):
    if pd.isna(val) or val == 'N/A':
        return 'unclassified'
    # Removes things like "(90%)" and extra spaces
    return str(val).split('(')[0].strip()

# ---------------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------------
def create_stability_plot(source_col, data):
    df = data.copy()
    # Apply cleaning to the target column
    df['classification'] = df[source_col].apply(clean_label)

    fig, ax = plt.subplots(figsize=(16, 8))
    
    # --- Hull Calculation ---
    def color_area(label, aplpha):
        points = df[df['classification'] == label][['flow_rate', 'voltage']].dropna().values
        if len(points) > 5: # Need a few points to make a meaningful hull
            x_log = np.log10(points[:, 0])
            y_raw = points[:, 1]
            
            x_min, x_max = x_log.min(), x_log.max()
            y_min, y_max = y_raw.min(), y_raw.max()
            if x_max == x_min or y_max == y_min: return

            # Normalize for alpha shape stability
            x_norm = (x_log - x_min) / (x_max - x_min)
            y_norm = (y_raw - y_min) / (y_max - y_min)
            points_norm = np.column_stack((x_norm, y_norm))

            try:
                alpha_val = aplpha 
                hull = alphashape.alphashape(points_norm, alpha_val)

                if hull.geom_type == 'Polygon':
                    geoms = [hull]
                elif hull.geom_type == 'MultiPolygon':
                    geoms = hull.geoms
                else:
                    return

                for poly in geoms:
                    coords = np.array(poly.exterior.coords)
                    real_x = 10**(coords[:, 0] * (x_max - x_min) + x_min)
                    real_y = coords[:, 1] * (y_max - y_min) + y_min
                    ax.add_patch(MplPolygon(np.column_stack((real_x, real_y)), 
                                            alpha=0.15, color=class_palette.get(label, '#000000'), 
                                            zorder=0, lw=0))
            except Exception as e:
                print(f"Hull error for {label} in {source_col}: {e}")

    # Draw hulls for main modes
    for mode in ['dripping', 'cone_jet', 'multi_jet', 'intermitent']:
        color_area(mode, 10.0)

    # --- Scatter Plot ---
    sns.scatterplot(
        data=df, x='flow_rate', y='voltage', hue='classification',
        palette=class_palette, alpha=0.8, edgecolor='k', s=70, zorder=2, ax=ax
    )

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}' if x >= 1 else f'{x}'))
    
    ax.set_xlabel('Flow Rate ($\\mu L/min$)')
    ax.set_ylabel('Voltage ($V$)')
    ax.set_title(f'{SOLUTION}\nElectrospray Stability Map\n(Classification Method: {source_col})')
    ax.legend(title='Classification', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    filename = f'data/plots/stability_map_{SOLUTION}_{source_col}.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
for source in PLOT_SOURCES:
    print(f"Generating plot for {SOLUTION}: {source}")
    create_stability_plot(source, df_raw)

print("\nDone. All stability maps have been generated.")