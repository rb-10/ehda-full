from feature_extraction import process_multiple_files
from ehda_normalization import prepare_training_data
from ehda_classifier import train

# 1. Load all data
df = process_multiple_files('*.json', folder=r'data/current_training')

# 2. Define the labels you want to EXCLUDE
exclude_labels = ['EXCLUDE', 'unconclusive', 'undefined', 'N/A']

# 3. Filter the DataFrame
df_clean = df[~df['label'].isin(exclude_labels)].copy()

# 4. VALIDATION (NEW!)
print("Total samples after filtering:", len(df_clean))
print("Remaining classes:\n", df_clean['label'].value_counts())

if len(df_clean) == 0:
    raise ValueError("No samples left after filtering!")

if df_clean['label'].nunique() < 2:
    raise ValueError("Need at least 2 classes!")

# Check for NaN in features
feature_cols = [c for c in df_clean.columns 
                if c not in ['sample_id', 'label', 'timestamp', 'source_file']]
if df_clean[feature_cols].isna().any().any():
    print("WARNING: Found NaN values, removing rows...")
    df_clean = df_clean.dropna()

# 5. Prepare training data
# Since we already filtered, use a non-existent exclude_label to skip double-filtering
df_raw, X, labels, feature_names, normalizer = prepare_training_data(
    df_clean, 
    drop_metadata=False,
    exclude_label="__FILTERED_ALREADY__"  # ← FIX: disable redundant filtering
)

# 6. Train models
results = train(X, labels, feature_names, normalizer=normalizer)