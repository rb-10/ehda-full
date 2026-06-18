import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Goes up 3 levels to 'main/'
sys.path.insert(0, str(project_root))

from mapping.software.database import ElectrosprayDatabase
from current_classification.train_code import build_feature_matrix, SAMPLING_FREQ, INVALID_LABELS
from current_classification.ehda_normalization import prepare_training_data

def create_feature_plots():
    BASE = project_root / "data"
    print(f"Loading database from {BASE}")
    db = ElectrosprayDatabase(str(BASE))
    
    # 1. Load Data
    df_db = db.load_training_dataframe()
    
    # 2. Filter Samples
    df_labeled = df_db[
        df_db['manual_classification'].notna() & 
        (~df_db['manual_classification'].isin(INVALID_LABELS))
    ].copy()
    
    print(f"Loaded {len(df_labeled)} samples with valid manual labels.")
    
    # 3. Build Matrix
    print("Extracting features...")
    df_features = build_feature_matrix(df_labeled, BASE / "raw_waveforms", SAMPLING_FREQ)
    
    # 4. Get raw features and normalizer
    _, _, labels, feature_names, normalizer = prepare_training_data(df_features)
    
    # 5. Fit normalizer and get normalized DataFrame
    print("Normalizing features...")
    df_normalized = normalizer.fit_transform(df_features)
    
    # Create output directories for the plots
    output_dir_raw = Path("current_classification/plots/raw")
    output_dir_norm = Path("current_classification/plots/normalized")
    output_dir_raw.mkdir(parents=True, exist_ok=True)
    output_dir_norm.mkdir(parents=True, exist_ok=True)
    
    # Set seaborn style for better visuals
    sns.set_theme(style="whitegrid")
    
    print(f"Generating plots for {len(feature_names)} features...")
    for feature in tqdm(feature_names):
        # Create a figure with 2 subplots (Raw and Normalized)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # We use a stripplot to see the clustering of samples for each label
        # 1. Raw Plot
        sns.stripplot(
            data=df_features, 
            x='label', 
            y=feature, 
            hue='label',
            ax=axes[0], 
            jitter=True, 
            alpha=0.7, 
            legend=False
        )
        axes[0].set_title(f"{feature} (Raw)")
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Normalized Plot
        sns.stripplot(
            data=df_normalized, 
            x='label', 
            y=feature, 
            hue='label',
            ax=axes[1], 
            jitter=True, 
            alpha=0.7, 
            legend=False
        )
        axes[1].set_title(f"{feature} (Normalized)")
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_dir_raw.parent / f"{feature}_comparison.png", dpi=150)
        plt.close(fig)
        
    print(f"Plots saved successfully in {output_dir_raw.parent.absolute()}")

if __name__ == "__main__":
    create_feature_plots()
