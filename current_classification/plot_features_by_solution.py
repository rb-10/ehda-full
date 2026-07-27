import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root and current directory to Python path
project_root = Path(__file__).parent.parent  # Goes up to 'main/'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from mapping.software.database import ElectrosprayDatabase
from current_classification.train_code import build_feature_matrix, INVALID_LABELS, compute_ratio_to_previous_step, resolve_label
from current_classification.ehda_normalization import prepare_training_data

# Standard electrospray modes sequence order
STANDARD_MODE_ORDER = [
    "dripping",
    "intermitent",
    "cone_jet",
    "multi_jet",
]

def create_feature_solution_plots():
    BASE = project_root / "data"
    print(f"Loading database from {BASE}")
    db = ElectrosprayDatabase(str(BASE))
    
    # 1. Load Data
    df_db = db.load_training_dataframe()
    if df_db.empty:
        print("No data loaded. Exiting.")
        sys.exit(0)
        
    # Backwards compatibility for DB columns
    if "sample_rate" not in df_db.columns:
        df_db["sample_rate"] = 100000.0
    if "n_samples" not in df_db.columns:
        df_db["n_samples"] = 50000

    df_db["sample_rate"] = df_db["sample_rate"].fillna(100000.0)
    df_db["n_samples"] = df_db["n_samples"].fillna(50000)

    df_db = compute_ratio_to_previous_step(df_db)
    df_db["final_label"] = df_db.apply(resolve_label, axis=1)
    
    # 2. Filter Samples with valid manual labels
    df_labeled = df_db[
        df_db['final_label'].notna() & 
        (~df_db['final_label'].isin(INVALID_LABELS))
    ].copy()
    
    print(f"Loaded {len(df_labeled)} samples with valid manual labels.")
    unique_solutions = list(df_labeled["solution_name"].dropna().unique())
    print(f"Solutions present in dataset ({len(unique_solutions)}): {unique_solutions}")
    
    all_features_dfs = []
    all_normalized_dfs = []

    # 3. Process each solution independently (handles different sample rates)
    for sol_name in unique_solutions:
        df_sol = df_labeled[df_labeled["solution_name"] == sol_name].copy()
        if df_sol.empty:
            continue

        sol_rates = df_sol["sample_rate"].unique()
        sol_sample_rate = float(sol_rates[0])
        print(f"\nProcessing solution '{sol_name}' ({len(df_sol)} samples, sample_rate = {sol_sample_rate} Hz)...")

        # Extract features for this solution using its specific sample rate
        df_sol_features = build_feature_matrix(df_sol, BASE / "raw_waveforms", sol_sample_rate)
        if df_sol_features.empty:
            print(f"Warning: No features extracted for solution '{sol_name}'. Skipping.")
            continue

        df_sol_features["solution_name"] = sol_name

        # Fit normalizer and get normalized DataFrame for this solution
        _, _, _, _, normalizer = prepare_training_data(df_sol_features)
        df_sol_norm = normalizer.fit_transform(df_sol_features)
        df_sol_norm["solution_name"] = sol_name

        all_features_dfs.append(df_sol_features)
        all_normalized_dfs.append(df_sol_norm)

    if not all_features_dfs:
        print("No feature data extracted across solutions. Exiting.")
        sys.exit(0)

    # 4. Superimpose/Combine DataFrames across solutions
    df_features = pd.concat(all_features_dfs, ignore_index=True)
    df_normalized = pd.concat(all_normalized_dfs, ignore_index=True)

    # Determine mode sequence order (standard order preserved, followed by any additional labels)
    present_labels = list(df_features['label'].dropna().unique())
    mode_order = [m for m in STANDARD_MODE_ORDER if m in present_labels] + [m for m in present_labels if m not in STANDARD_MODE_ORDER]
    print(f"\nMode sequence order for plots: {mode_order}")

    # Determine feature names (common numerical columns across solutions)
    _, _, _, feature_names, _ = prepare_training_data(df_features)

    # Create output directory for the solution comparison plots
    output_dir = Path("current_classification/plots/comparison_by_solution")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seaborn style for clean, modern visuals
    sns.set_theme(style="whitegrid")
    
    print(f"Generating solution-superimposed plots for {len(feature_names)} features...")
    for feature in tqdm(feature_names):
        # Create a figure with 2 subplots (Raw and Normalized)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # 1. Raw Plot with matched mode sequence (order) subdivided by solution_name (hue)
        sns.boxplot(
            data=df_features,
            x='label',
            y=feature,
            hue='solution_name',
            order=mode_order,
            ax=axes[0],
            showfliers=False,
            boxprops=dict(alpha=0.3),
            legend=False
        )
        sns.stripplot(
            data=df_features, 
            x='label', 
            y=feature, 
            hue='solution_name',
            order=mode_order,
            ax=axes[0], 
            dodge=True,
            jitter=0.2, 
            alpha=0.7,
            linewidth=0.5
        )
        axes[0].set_title(f"{feature} (Raw - Grouped by Solution)", fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Deduplicate legend for subplot 0
        handles, labels_leg = axes[0].get_legend_handles_labels()
        n_sols = len(df_features['solution_name'].unique())
        axes[0].legend(handles[:n_sols], labels_leg[:n_sols], title="Solution", bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # 2. Normalized Plot with matched mode sequence (order) subdivided by solution_name (hue)
        sns.boxplot(
            data=df_normalized,
            x='label',
            y=feature,
            hue='solution_name',
            order=mode_order,
            ax=axes[1],
            showfliers=False,
            boxprops=dict(alpha=0.3),
            legend=False
        )
        sns.stripplot(
            data=df_normalized, 
            x='label', 
            y=feature, 
            hue='solution_name',
            order=mode_order,
            ax=axes[1], 
            dodge=True,
            jitter=0.2, 
            alpha=0.7,
            linewidth=0.5
        )
        axes[1].set_title(f"{feature} (Normalized - Grouped by Solution)", fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Deduplicate legend for subplot 1
        handles1, labels_leg1 = axes[1].get_legend_handles_labels()
        axes[1].legend(handles1[:n_sols], labels_leg1[:n_sols], title="Solution", bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_dir / f"{feature}_solution_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print(f"\nPlots saved successfully in: {output_dir.absolute()}")

if __name__ == "__main__":
    create_feature_solution_plots()
