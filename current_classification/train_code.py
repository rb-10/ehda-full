# pip install numpy pandas scikit-learn scipy matplotlib seaborn xgboost pywt
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Goes up 3 levels to 'main/'
sys.path.insert(0, str(project_root))


from mapping.software.database import ElectrosprayDatabase # Import your DB class
from mapping.software.electrospray import ElectrosprayDataProcessing
from ehda_normalization import prepare_training_data
from ehda_classifier import train


SAMPLING_FREQ = 1e5
RECORD_LENGTH = 50_000
MULTIPLIER_NA = 1.0 # Data save in DB already in nA, no need to multiply
CUTOFF_HZ     = 3_000


EXCLUDE_FEATURES = [
    "actual_current_ps", # If you find it's too noisy
    "wt_detail_L1_energy", # Often just high-frequency noise
    "wt_detail_L1_energy_rel"
]
# List labels you want to ignore/drop
INVALID_LABELS = ["undefined", "unknown", "noise", "", None]

def build_feature_matrix(df_db, raw_dir, sample_rate):
    """
    Iterates through DB records, loads raw .npy files, and uses the 
    ElectrosprayDataProcessing class to generate the 66-feature vector.
    """
    processing = ElectrosprayDataProcessing(sample_rate)
    
    # Filter setup (same as live)
    cutoff = CUTOFF_HZ / (0.5 * sample_rate)
    b, a = butter(6, Wn=cutoff, btype="low", analog=False)
    
    all_rows = []
    
    print(f"Extracting features from {len(df_db)} samples...")
    for _, row in tqdm(df_db.iterrows(), total=len(df_db)):
        file_path = Path(raw_dir) / str(row['raw_data_file'])
        
        if not file_path.exists():
            continue
            
        try:
            # 1. Load and Clear
            datapoints = np.load(file_path) * MULTIPLIER_NA
            processing.clear_results()
            
            # 2. Process (Matches live acquire_and_process logic)
            processing.calculate_filter(a, b, datapoints)
            processing.calculate_statistics(processing.datapoints_filtered)
            processing.calculate_power_spectral_density(processing.datapoints_filtered)
            processing.extract_advanced_ml_features()
            
            # 3. Harvest Features
            feats = processing.get_db_features_dictionary() # mean_na, etc.
            feats.update(processing.ml_features)            # advanced stats
            
            # 4. Add Metadata (Matches live classify_sample logic)
            feats.update({
                "target_voltage": float(row["target_voltage"]),
                "actual_voltage": float(row["actual_voltage"]),
                "flow_rate":      float(row["flow_rate"]),
                "voltage_error":  float(row["actual_voltage"]) - float(row["target_voltage"]),
                "current_PS":     float(row.get("actual_current_ps", 0.0)),
                "label":          row["image_classification"] or row["manual_classification"]
            })
            
            all_rows.append(feats)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    df = pd.DataFrame(all_rows)
    # 3. DROP SPECIFIC FEATURES
    # We do this before normalization so the normalizer doesn't look for them
    df = df.drop(columns=[c for c in EXCLUDE_FEATURES if c in df.columns])        
    return df


# 0 - Init DB
BASE = Path(r"C:\Users\HV\Desktop\bruno_work\main\data")
db = ElectrosprayDatabase(str(BASE))


# 1. Load Data
df_db = db.load_training_dataframe()

# 2. FILTER SAMPLES: Keep only rows with valid manual labels
# This removes NaN values and values in your INVALID_LABELS list
df_labeled = df_db[
    df_db['manual_classification'].notna() & 
    (~df_db['manual_classification'].isin(INVALID_LABELS))
].copy()

print(f"Training on {len(df_labeled)} samples with valid manual labels.")

# 3. Build Matrix
df_features = build_feature_matrix(df_labeled, BASE / "raw_waveforms", SAMPLING_FREQ)

# 4. Normalize and Train
# (The normalizer will now only see the columns remaining in df_features)
df_norm, X, labels, feature_names, normalizer = prepare_training_data(
    df_features, scaler_save_path="scalers"
)

results = train(X, labels, feature_names, save_folder="models")