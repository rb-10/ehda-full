import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import alphashape
from matplotlib.patches import Polygon as MplPolygon

# ---------------------------------------------------------
# SETTINGS - CHANGE SOURCE HERE
# ---------------------------------------------------------
# Options: 'rf_spray_mode', 'xgb_spray_mode', or 'final_class', 'manual_classification'
PLOT_SOURCE = 'raw_class' 
FILE_PATH = r'C:\Users\HV\Desktop\bruno_work\save_electrospray\DMF\Current\cleaned_classification_results_20260422_094456.csv'

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv(FILE_PATH)

# Map the chosen source to a standard column for the script
df['classification'] = df[PLOT_SOURCE]

# Set up color palette
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

plt.figure(figsize=(16, 8))
ax = plt.gca()

# ---------------------------------------------------------
# ROBUST CONCAVE HULL CALCULATION
# ---------------------------------------------------------
def color_area(classification_label):
    # Filter points for the specific label
    points = df[df['classification'] == classification_label][['flow_rate', 'voltage']].dropna().values

    if len(points) > 3:
        # Scale data for log-linear calculation
        x_log = np.log10(points[:, 0])
        y_raw = points[:, 1]
        
        x_min, x_max = x_log.min(), x_log.max()
        y_min, y_max = y_raw.min(), y_raw.max()
        
        if x_max == x_min or y_max == y_min: return

        # Normalize points to [0, 1] for stable Alpha Shape
        x_norm = (x_log - x_min) / (x_max - x_min)
        y_norm = (y_raw - y_min) / (y_max - y_min)
        points_norm = np.column_stack((x_norm, y_norm))

        try:
            alpha_val = 2.0 
            hull = alphashape.alphashape(points_norm, alpha_val)

            def add_hull_to_plot(geometry):
                if geometry.geom_type == 'Polygon':
                    coords = np.array(geometry.exterior.coords)
                    # De-normalize
                    real_x = 10**(coords[:, 0] * (x_max - x_min) + x_min)
                    real_y = coords[:, 1] * (y_max - y_min) + y_min
                    ax.add_patch(MplPolygon(np.column_stack((real_x, real_y)), 
                                            alpha=0.15, color=class_palette.get(classification_label, '#000000'), 
                                            zorder=0, lw=0))
                elif geometry.geom_type == 'MultiPolygon':
                    for part in geometry.geoms:
                        add_hull_to_plot(part)

            add_hull_to_plot(hull)
        except Exception as e:
            print(f"Hull error for {classification_label}: {e}")

for mode in ['dripping', 'cone_jet', 'multi_jet', 'intermitent']:
    color_area(mode)

# ---------------------------------------------------------
# PROCEED WITH PLOTTING
# ---------------------------------------------------------
sns.scatterplot(
    data=df, x='flow_rate', y='voltage', hue='classification',
    palette=class_palette, alpha=0.8, edgecolor='k', s=70, zorder=2
)

plt.xscale('log')

# Formatter for log scale ticks
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}' if x >= 1 else f'{x}'))

plt.xlabel('Flow Rate ($\mu L/min$)')
plt.ylabel('Voltage ($V$)')
plt.title(f'Electrospray Stability Map (Source: {PLOT_SOURCE})')
plt.legend(title='Classification', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig(f'plot_{PLOT_SOURCE}.png')
plt.show()
