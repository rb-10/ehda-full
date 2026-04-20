import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import alphashape
from matplotlib.patches import Polygon as MplPolygon

# Load the CSV file
df = pd.read_csv('extras/classification_data.csv', header=None)
df.columns = ['voltage', 'flow_rate', 'classification', 'source', 'experiment']

# Set up color palette
class_palette = {
    'dripping': "#fffb00",
    'intermitent': "#0059ff",
    'cone_jet': "#009b00",
    'multi_jet': "#ff0000",
    'undefined': "#525252",
    'unclassified': "#00FFDD",
    'unconclusive': "#000000",
    'EXCLUDE': '#7f7f7f',
}

plt.figure(figsize=(16, 8))
ax = plt.gca()

# ---------------------------------------------------------
# ROBUST CONCAVE HULL CALCULATION
# ---------------------------------------------------------
def color_area(classification):

    # 1. Filter cone_jet points
    cj_points = df[df['classification'] == classification][['flow_rate', 'voltage']].dropna().values

    if len(cj_points) > 2:
        # 2. SCALE DATA for calculation (Log X, Linear Y)
        # We use Log10 for flow_rate because your plot uses a log scale
        x_log = np.log10(cj_points[:, 0])
        y_raw = cj_points[:, 1]
        
        # Normalize both to [0, 1] range so alpha parameter is predictable
        x_min, x_max = x_log.min(), x_log.max()
        y_min, y_max = y_raw.min(), y_raw.max()
        
        # Normalized points
        x_norm = (x_log - x_min) / (x_max - x_min)
        y_norm = (y_raw - y_min) / (y_max - y_min)
        points_norm = np.column_stack((x_norm, y_norm))

        try:
            # 3. Generate Alpha Shape on normalized data
            # alpha=0 is Convex Hull. Increase it (e.g., 2.0, 5.0) for more concavity.
            alpha_val = 20.0 
            hull = alphashape.alphashape(points_norm, alpha_val)

            # 4. DE-NORMALIZE and Plot
            def add_hull_to_plot(geometry):
                if geometry.geom_type == 'Polygon':
                    coords = np.array(geometry.exterior.coords)
                    # Reverse the normalization and the log
                    real_x = 10**(coords[:, 0] * (x_max - x_min) + x_min)
                    real_y = coords[:, 1] * (y_max - y_min) + y_min
                    ax.add_patch(MplPolygon(np.column_stack((real_x, real_y)), 
                                            alpha=0.2, color=class_palette[classification], 
                                            zorder=0, lw=0))
                elif geometry.geom_type == 'MultiPolygon':
                    for part in geometry.geoms:
                        add_hull_to_plot(part)

            add_hull_to_plot(hull)
            print(f"Shape successfully drawn using alpha={alpha_val}")

        except Exception as e:
            print(f"Shape error: {e}")


color_area('dripping')
color_area('cone_jet')
color_area('multi_jet')
color_area('intermitent')

# ---------------------------------------------------------
# PROCEED WITH PLOTTING
# ---------------------------------------------------------
sns.scatterplot(
    data=df, x='flow_rate', y='voltage', hue='classification',
    palette=class_palette, alpha=0.7, edgecolor='k', s=60, zorder=2
)

plt.xscale('log')
# Set custom ticks for log scale
min_x = max(1, int(df['flow_rate'].min()))
max_x = int(df['flow_rate'].max())
major_ticks = [1, 10, 100, 1000]
intermediate_ticks = list(range(10, 80, 10))
ticks = list(range(1, 8, 1))
ticks2 = list(range(10, 15, 2))
all_ticks = sorted(set(major_ticks + intermediate_ticks + ticks + ticks2))
all_ticks = [tick for tick in all_ticks if min_x <= tick <= max_x]
ax.set_xticks(all_ticks)
ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}' if x >= 1 else ''))

plt.xlabel('Flow Rate(uL/min)')
plt.ylabel('Voltage(V)')
plt.title('DMF Classification Map: Flow Rate vs Voltage')
plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()