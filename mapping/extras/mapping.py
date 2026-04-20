import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIG ---
JSON_FOLDER = Path(r"C:\Users\HV\Desktop\bruno_work\save_electrospray\DMF\Current")  # Change as needed

# Color mapping for classes
CLASS_COLORS = {
	"cone_jet": "green",
	"dripping": "blue",
	"intermitent": "orange",
	"multi_jet": "red",
	"unconclusive": "gray",
	"undefined": "black"
}

voltages = []
flow_rates = []
colors = []
labels = []

for json_file in JSON_FOLDER.glob("experiment_*.json"):
	with open(json_file, "r") as f:
		data = json.load(f)
	for sample_key, sample in data.items():
		# Try to get voltage, flow rate, and classification
		voltage = sample.get("Voltage") or sample.get("voltage")
		flow_rate = sample.get("Flow Rate") or sample.get("flow_rate")
		classification = sample.get("image_classification")
		if voltage is None or flow_rate is None or classification is None:
			continue
		voltages.append(voltage)
		flow_rates.append(flow_rate)
		colors.append(CLASS_COLORS.get(classification, "gray"))
		labels.append(classification)

plt.figure(figsize=(8,6))
scatter = plt.scatter(flow_rates, voltages, c=colors, alpha=0.7)	
plt.xscale('log')

# Create legend manually
import matplotlib.patches as mpatches
legend_handles = [mpatches.Patch(color=color, label=cls) for cls, color in CLASS_COLORS.items()]
plt.legend(handles=legend_handles, title="Classification")

plt.ylabel("Voltage (V)")
plt.xlabel("Flow Rate (uL/min)")
plt.title("Electrospray Mode Map: (Reviewed)")
plt.grid(True)
plt.tight_layout()
plt.show()
