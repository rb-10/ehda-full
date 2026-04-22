import os
import json
import pandas as pd
import re

FOLDER = r"data\EW82_HV_nz_21-04\Current"
CSV = r"classification_results_20260421_143641.csv"
CONFIDENCE = 0.7

def clean_and_threshold_mode(mode_string, threshold=0.70):
    """
    1. If percentage < threshold -> 'unconclusive'
    2. If percentage >= threshold -> remove '(XX%)'
    """
    if not isinstance(mode_string, str):
        return mode_string
    
    # Regex to find the label and the percentage
    # Group 1: The label (e.g., dripping)
    # Group 2: The percentage number (e.g., 89)
    match = re.search(r'^(.*?)\s*\((\d+)%\)', mode_string)
    
    if match:
        label = match.group(1).strip()
        percentage = int(match.group(2)) / 100.0
        
        if percentage < threshold:
            return "unconclusive"
        else:
            return label # Returns just 'dripping' or 'intermitent'
            
    return mode_string

def process_electrospray(folder_path, csv_name, threshold=0.70):
    csv_path = os.path.join(folder_path, csv_name)
    if not os.path.exists(csv_path):
        print(f"File {csv_name} not found.")
        return

    df = pd.read_csv(csv_path)

    # Dictionary to store JSON data indexed by (exp_idx, sample_idx)
    json_updates = {}

    for f in os.listdir(folder_path):
        if f.endswith(".json"):
            with open(os.path.join(folder_path, f), 'r') as file:
                data = json.load(file)
                exp_idx = data.get("_meta", {}).get("experiment_index")
                
                for key, content in data.items():
                    if key.startswith("sample "):
                        s_idx = int(re.search(r'\d+', key).group())
                        
                        # Clean both modes using the new logic
                        rf_cleaned = clean_and_threshold_mode(content.get('rf_spray_mode'), threshold)
                        xgb_cleaned = clean_and_threshold_mode(content.get('xgb_spray_mode'), threshold)
                        manual = content.get('manual_classification')
                        json_updates[(exp_idx, s_idx)] = {
                            'voltage': content.get('voltage'),
                            'flow_rate': content.get('flow_rate'),
                            'rf_spray_mode': rf_cleaned,
                            'xgb_spray_mode': xgb_cleaned,
                            'manual_classification' : manual
                        }

    # Apply updates to the existing DataFrame
    for i, row in df.iterrows():
        key = (row['experiment_idx'], row['sample_idx'])
        if key in json_updates:
            df.at[i, 'voltage'] = json_updates[key]['voltage']
            df.at[i, 'flow_rate'] = json_updates[key]['flow_rate']
            df.at[i, 'rf_spray_mode'] = json_updates[key]['rf_spray_mode']
            df.at[i, 'xgb_spray_mode'] = json_updates[key]['xgb_spray_mode']
            df.at[i, 'manual_classification'] = json_updates[key]['manual_classification']

    # Save the cleaned file
    output_path = os.path.join(folder_path, f"cleaned_{csv_name}")
    df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to: {output_path}")

# Example Usage:
# process_electrospray_json_to_csv('./data', 'experiment_results.csv', threshold=0.70)

if __name__ == "__main__":
    process_electrospray(FOLDER, CSV, CONFIDENCE)