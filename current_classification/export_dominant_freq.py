import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mapping.software.database import ElectrosprayDatabase
from current_classification.train_code import build_feature_matrix, SAMPLING_FREQ, INVALID_LABELS
from current_classification.ehda_normalization import prepare_training_data

def export_dominant_freq():
    BASE = project_root / "data"
    print(f"Loading database from {BASE}")
    db = ElectrosprayDatabase(str(BASE))
    
    df_db = db.load_training_dataframe()
    df_labeled = df_db[
        df_db['manual_classification'].notna() & 
        (~df_db['manual_classification'].isin(INVALID_LABELS))
    ].copy()
    
    print("Extracting features (this will take a moment)...")
    df_features = build_feature_matrix(df_labeled, BASE / "raw_waveforms", SAMPLING_FREQ)
    _, _, labels, feature_names, normalizer = prepare_training_data(df_features)
    
    print("Normalizing features...")
    df_normalized = normalizer.fit_transform(df_features)
    
    # Extract dominant_freq from both raw and normalized DataFrames
    export_df = pd.DataFrame({
        'sample_index': df_features.index,
        'label': df_features['label'],
        'dominant_freq_raw': df_features['dominant_freq'],
        'dominant_freq_normalized': df_normalized['dominant_freq']
    })
    
    output_path = Path("current_classification/plots") / "dominant_freq_values.csv"
    export_df.to_csv(output_path, index=False)
    print(f"Successfully exported dominant_freq values to {output_path.absolute()}")

if __name__ == "__main__":
    export_dominant_freq()
