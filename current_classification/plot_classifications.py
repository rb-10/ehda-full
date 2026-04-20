import os
import json
import matplotlib.pyplot as plt

# Folder containing your JSON files
FOLDER_PATH = r'C:\Users\HV\Desktop\bruno_work\save_electrospray\DMF\test'

# Define a color palette for the different classification modes
COLOR_MAP = {
    "cone_jet": "green",
    "dripping": "blue",
    "intermittent": "orange",
    "multi_jet": "purple",
    "corona": "red",
    "undefined": "gray",
    "unconclusive": "maroon"
}

def create_and_save_plot(model_name, points_data, title, output_filename):
    """ Helper function to plot and save a mapping for a specific analytical model """
    plt.figure(figsize=(10, 7))

    plotted_any = False
    for mode, coords in points_data.items():
        if len(coords["x"]) > 0:
            plotted_any = True
            plt.scatter(
                coords["x"], 
                coords["y"], 
                c=COLOR_MAP.get(mode, "black"), 
                label=mode, 
                alpha=0.8,
                edgecolors='w',
                linewidth=0.5,
                s=60
            )

    if not plotted_any:
        print(f"No valid data to plot for {model_name}!")
        plt.close()
        return

    # Add labels, title, and legend
    plt.title(title)
    plt.xlabel("Flow Rate")
    plt.ylabel("Voltage (V)")
    
    # Add a soft log scale ("not too intense") to the flow rate x-axis
    plt.xscale('symlog', linthresh=1.0)
    
    plt.legend(title="Spray Mode", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust layout to fit legend
    plt.tight_layout()
    
    # Save the plot instead of showing it
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot: {output_filename}")
    plt.close()

def main():
    if not os.path.exists(FOLDER_PATH):
        print(f"Folder not found: {FOLDER_PATH}")
        return

    print(f"Reading JSONs from {FOLDER_PATH}...")

    # The 3 models you're tracking in your json files
    model_keys = ["classical_model", "ml_model", "nn_model", "image_classification"]
    
    # Set up empty dictionaries to hold coordinate points for each of the models
    all_models_data = {
        model_key: {mode: {"x": [], "y": []} for mode in COLOR_MAP.keys()}
        for model_key in model_keys
    }

    # Iterate through all JSON files in the directory
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('.json'):
            json_path = os.path.join(FOLDER_PATH, filename)
            
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Could not read {filename}: {e}")
                    continue

            for key, value in data.items():
                if key.startswith('sample') and isinstance(value, dict):
                    voltage = value.get('voltage')
                    flow_rate = value.get('flow_rate')
                    
                    if voltage is None or flow_rate is None:
                        continue
                        
                    # Extract the classification for all 3 models!
                    for model_key in model_keys:
                        mode = value.get(model_key, 'Undefined')
                        
                        # Sometimes models may be saved as a list if we accidentally serialized an array:
                        if isinstance(mode, list) and len(mode) > 0:
                            mode = mode[0]
                        if mode is None:
                            mode = 'Undefined'
                            
                        # Standardize strings
                        mode = str(mode).strip().strip("'").strip('"')

                        # Keep the script stable by creating new sub-dicts if unknown modes pop up somehow
                        if mode not in all_models_data[model_key]:
                            if mode not in COLOR_MAP:
                                COLOR_MAP[mode] = "black"
                            all_models_data[model_key][mode] = {"x": [], "y": []}
                        
                        all_models_data[model_key][mode]["x"].append(flow_rate) # X Axis
                        all_models_data[model_key][mode]["y"].append(voltage)   # Y Axis

    # Defining pretty titles for each graph
    titles = {
        "classical_model": "Electrospray Mode Map (Classical Classification)",
        "ml_model": "Electrospray Mode Map (Machine Learning Classification)",
        "nn_model": "Electrospray Mode Map (Neural Network Classification)",
        "image_classification": "Electrospray Mode Map (Image Classification)"
    }

    # Automatically generate, color, and save a high-res plot for each analytical methodology
    for model_key in model_keys:
        output_filename = os.path.join(FOLDER_PATH, f"Map_{model_key}.png")
        create_and_save_plot(
            model_name=model_key, 
            points_data=all_models_data[model_key], 
            title=titles[model_key], 
            output_filename=output_filename
        )

if __name__ == '__main__':
    main()
