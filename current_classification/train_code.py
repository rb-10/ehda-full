# pip install numpy pandas scikit-learn scipy matplotlib seaborn xgboost pywt
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter
from tqdm import tqdm
import re

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
    "current_PS",
    "voltage_error",
    "target_voltage",
    "actual_voltage",
    "mean_na",
    "median_na",
    "rms_na"
]
# List labels you want to ignore/drop
INVALID_LABELS = ["N/A","undefined","unconclusive","noise", "", None]

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
            processing.extract_advanced_ml_features(ratio_to_previous_step=float(row["ratio_to_previous_step"]))
            
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
                "label":          row["final_label"]
            })
            
            all_rows.append(feats)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    df = pd.DataFrame(all_rows)
    # 3. DROP SPECIFIC FEATURES
    # We do this before normalization so the normalizer doesn't look for them
    df = df.drop(columns=[c for c in EXCLUDE_FEATURES if c in df.columns])        
    return df

def clean_label(label):
    """
    Strips confidence annotations like '(98%)' from image_classification labels,
    normalizes whitespace/case, and returns None for empty/invalid values.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    label = str(label)
    # Remove anything in parentheses, e.g. "cone_jet (98%)" -> "cone_jet"
    label = re.sub(r"\(.*?\)", "", label).strip()
    return label if label else None

def resolve_label(row):
    """
    Returns manual_classification if it's valid, otherwise falls back to
    a cleaned-up image_classification label.
    """
    manual = row.get("manual_classification")
    if pd.notna(manual) and manual not in INVALID_LABELS:
        return manual

    fallback = clean_label(row.get("image_classification"))
    return fallback  # could still be None/invalid, filtered out later

def compute_ratio_to_previous_step(df_db):
    """
    Vectorized equivalent of database.get_previous_step_mean_na(), computed
    over the full historical dataframe instead of one DB query per row.

    Assumes df_db is ordered by `id` ascending (matches insertion/time order).
    A "step" is a contiguous run of rows sharing the same target_voltage.
    For each row, looks at the previous step (the run of rows immediately
    before the current step began) and, if that previous step's flow_rate
    matches the current row's flow_rate, sets
    ratio = current mean_na / mean(previous step's mean_na).
    Otherwise (no previous step, or flow_rate differs) ratio = 1.0.
    """
    df = df_db.sort_values("id").reset_index(drop=True)

    # Assign a step id: increments every time target_voltage changes
    step_id = (df["target_voltage"] != df["target_voltage"].shift()).cumsum()

    step_mean_na = df.groupby(step_id)["mean_na"].mean()
    step_flow    = df.groupby(step_id)["flow_rate"].first()

    unique_steps = step_id.drop_duplicates().tolist()
    ratios = np.ones(len(df))

    for i, sid in enumerate(unique_steps):
        mask = (step_id == sid).values
        if i == 0:
            continue  # no earlier step -> ratio stays 1.0

        prev_sid = unique_steps[i - 1]
        prev_mean = step_mean_na.loc[prev_sid]
        prev_flow = step_flow.loc[prev_sid]
        cur_flow  = step_flow.loc[sid]

        if cur_flow == prev_flow and pd.notna(prev_mean) and prev_mean != 0:
            ratios[mask] = df.loc[mask, "mean_na"].values / prev_mean
        # else leave as 1.0 (flow_rate mismatch or invalid previous mean)

    df["ratio_to_previous_step"] = ratios
    return df
if __name__ == '__main__':
    # 0 - Init DB
    BASE = Path(r"data")
    print(os.getcwd())
    db = ElectrosprayDatabase(str(BASE))
        
    
    # 1. Load Data
    df_db = db.load_training_dataframe()
    df_db = compute_ratio_to_previous_step(df_db)
    df_db["final_label"] = df_db.apply(resolve_label, axis=1)

    # 2. FILTER SAMPLES: Keep only rows with valid manual labels
    # This removes NaN values and values in your INVALID_LABELS list
    df_labeled = df_db[
        df_db['final_label'].notna() & 
        (~df_db['final_label'].isin(INVALID_LABELS))
    ].copy()
    
    print(f"Training on {len(df_labeled)} samples with valid manual labels.")
    
    # 3. Build Matrix
    df_features = build_feature_matrix(df_labeled, BASE / "raw_waveforms", SAMPLING_FREQ)
    
    # 4. Normalize and Train
    # A. Prepare data using your custom normalization logic
    df_norm, X, labels, feature_names, normalizer = prepare_training_data(df_features)
    
    # B. Pass the UN-FITTED normalizer into your original train function
    # The function will then fit it on the training split as you intended.
    training_results = train(
        X=X, 
        labels=labels, 
        feature_names=feature_names, 
        normalizer=normalizer, 
        save_folder="current_classification/models"
    )
