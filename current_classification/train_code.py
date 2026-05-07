# pip install numpy pandas scikit-learn scipy matplotlib seaborn xgboost pywt
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Goes up 3 levels to 'main/'
sys.path.insert(0, str(project_root))


from mapping.software.database import ElectrosprayDatabase # Import your DB class
from feature_extraction import build_training_dataframe
from ehda_normalization import prepare_training_data
from ehda_classifier import train

# 0 - Init DB
BASE = Path(r"C:\Users\HV\Desktop\bruno_work\main\data")
db = ElectrosprayDatabase(str(BASE))


# 1. Load all data - Path to databse
df = db.load_training_dataframe()
#Columns: id, timestamp, solution_name, hv_position, target_voltage, actual_voltage, actual_current_ps, flow_rate, mean_na, deviation_na, median_na, rms_na, variance_na, qty_max, pct_max, band_power_v_low, band_power_low, band_power_mid, band_power_high, band_power_v_high, rf_spray_mode, xgb_spray_mode, image_classification, manual_classification, video_file, raw_data_file 
print(f"Loaded {len(df)} samples from database")
print(f"\nAvailable columns:")
for col in df.columns:
    print(f"  - {col}")


# 2. Clean data
# Define labels to exclude
exclude_labels = ['EXCLUDE', 'unconclusive', 'undefined', 'N/A']

# Clean the DataFrame
df_clean = df[~df['manual_classification'].isin(exclude_labels)].copy()

# Validation
print(f"Total samples after cleaning: {len(df_clean)}")
print(f"\nRemaining classes:")
print(df_clean['manual_classification'].value_counts())

if len(df_clean) == 0:
    raise ValueError("No samples left after cleaning!")

if df_clean['label'].nunique() < 2:
    raise ValueError("Need at least 2 classes!")
    

# 4. Extract Features from Raw Waveforms    
raw_waveforms_dir = BASE / "raw_waveforms"
if not raw_waveforms_dir.exists():
    raise FileNotFoundError(
        f"Raw waveforms directory not found: {raw_waveforms_dir}\n"
        f"Make sure your .npy files are stored in this location."
    )

# Build feature DataFrame
# This function:
# 1. Loads each .npy file
# 2. Applies the same filtering as live pipeline
# 3. Calculates statistics, FFT, band powers
# 4. Extracts advanced features (wavelet, spectral, shape)

df_features = build_training_dataframe(df_clean, raw_waveforms_dir)
if len(df_features) == 0:
    raise ValueError(
        "No features extracted! Check that:\n"
        "   1. .npy files exist in raw_waveforms directory\n"
        "   2. 'raw_data_file' column in database matches .npy filenames"
    )

# 5. Prepare training data
# Since we already filtered, use a non-existent exclude_label to skip double-filtering
df_raw, X, labels, feature_names, normalizer = prepare_training_data(
    df_clean, 
    drop_metadata=False,
    exclude_label="__FILTERED_ALREADY__"  # ← FIX: disable redundant filtering
)

# 6. Train models
results = train(X, labels, feature_names, normalizer=normalizer)