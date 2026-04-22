import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIG ---
JSON_FOLDER = Path(r"data\DMF\Current")

CLASS_COLORS = {
    "cone_jet": "green",
    "dripping": "blue",
    "intermitent": "orange",  # keep your spelling if consistent in data
    "multi_jet": "purple",
    "corona": "red",
    "undefined": "gray",
    "unconclusive": "maroon"
}

# Store grouped data (like your first script)
grouped_data = {
    cls: {"x": [], "y": []} for cls in CLASS_COLORS.keys()
}

# --- LOAD DATA ---
for json_file in JSON_FOLDER.glob("experiment_*.json"):
    with open(json_file, "r") as f:
        data = json.load(f)

    for sample_key, sample in data.items():
        voltage = sample.get("Voltage") or sample.get("voltage")
        flow_rate = sample.get("Flow Rate") or sample.get("flow_rate")
        classification = sample.get("image_classification")

        if voltage is None or flow_rate is None or classification is None:
            continue

        classification = str(classification).strip().lower()

        # Handle unexpected classes safely
        if classification not in grouped_data:
            grouped_data[classification] = {"x": [], "y": []}
            if classification not in CLASS_COLORS:
                CLASS_COLORS[classification] = "black"

        grouped_data[classification]["x"].append(flow_rate)
        grouped_data[classification]["y"].append(voltage)

# --- PLOT ---
plt.figure(figsize=(10, 7))

plotted_any = False

for cls, coords in grouped_data.items():
    if len(coords["x"]) > 0:
        plotted_any = True
        plt.scatter(
            coords["x"],
            coords["y"],
            c=CLASS_COLORS.get(cls, "black"),
            label=cls,
            alpha=0.8,
            edgecolors='w',
            linewidth=0.5,
            s=60
        )

if not plotted_any:
    print("No valid data to plot!")
    exit()

# Axis settings
plt.xscale('log')  # your requested log scale
plt.xlabel("Flow Rate (uL/min)")
plt.ylabel("Voltage (V)")
plt.title("Electrospray Mode Map: (Image Model)")

# Legend
plt.legend(title="Spray Mode", bbox_to_anchor=(1.05, 1), loc='upper left')

# Grid + layout
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()