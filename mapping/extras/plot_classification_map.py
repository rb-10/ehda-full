import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import alphashape
from matplotlib.patches import Polygon as MplPolygon

from mapping.software.database import ElectrosprayDatabase

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
BASE = Path(r"C:\Users\HV\Desktop\bruno_work\main\data")
DB_PATH = str(BASE)
SOLUTION = "1KHZ_TEST1"
COMPARISON_MAP = True

PLOT_SOURCES = [
    'image_classification',
    'ml_classification',
    'nn_classification',
    'generalist_ml_classification',
    'rf_spray_mode',
    'xgb_spray_mode',
    'classical_classification',
]

class_palette = {
    'dripping':      "#2e62d4",
    'intermitent':   "#066400",
    'cone_jet':      "#e80101",
    'multi_jet':     "#830068",
    'undefined':     "#ffa700",
    'unclassified':  "#ffa700",
    'unconclusive':  "#ffa700",
    'EXCLUDE':       '#7f7f7f',
    'none':          '#7f7f7f',
    'corona':        '#e200b2',
}

# ---------------------------------------------------------
# DATA LOADING & DIAGNOSTICS
# ---------------------------------------------------------
print("Opening DB …")
db = ElectrosprayDatabase(DB_PATH)

# ── FIX 2a: see every solution name that actually exists ──────────────────────
all_solutions = pd.read_sql(
    "SELECT DISTINCT solution_name FROM measurements", db._conn
)
print("Solution names found in DB:")
print(all_solutions.to_string(index=False))

# ── FIX 2b: normalise SOLUTION (strip whitespace, consistent case) ────────────
SOLUTION_NORM = SOLUTION.strip()          # add .upper() here AND in the DB values if needed

query = f"SELECT * FROM measurements WHERE TRIM(solution_name) = '{SOLUTION_NORM}'"
df_raw = pd.read_sql(query, db._conn)

print(f"\nClosed DB.  Rows returned for '{SOLUTION_NORM}': {len(df_raw)}")

# ── FIX 2c: fail early with a clear message instead of silent empty plots ─────
if df_raw.empty:
    db.close()
    raise RuntimeError(
        f"Query returned 0 rows for solution '{SOLUTION_NORM}'.\n"
        "Check the solution names printed above – look for extra spaces or "
        "different capitalisation."
    )

db.close()
print(f"DataFrame shape: {df_raw.shape}")
print(df_raw.head(3))

# ---------------------------------------------------------
# FIX 1: only keep sources whose column actually exists in the DataFrame
# ---------------------------------------------------------
available_sources = [s for s in PLOT_SOURCES if s in df_raw.columns]
skipped = set(PLOT_SOURCES) - set(available_sources)
if skipped:
    print(f"\nSkipping sources not present in this DB: {sorted(skipped)}")
print(f"Sources to plot: {available_sources}\n")

# ---------------------------------------------------------
# LABEL CLEANING
# ---------------------------------------------------------
def clean_label(val):
    if pd.isna(val) or val == 'N/A':
        return 'unclassified'
    return str(val).split('(')[0].strip()

# ---------------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------------
def create_stability_plot(source_col, data):
    df = data.copy()
    df['classification'] = df[source_col].apply(clean_label)

    # --- Aggregate duplicate (flow_rate, actual_voltage) points ---
    if source_col == 'image_classification':
        # Pick the non-unclassified label; fall back to 'unclassified' if all are
        def image_agg(labels):
            non_unc = [l for l in labels if l != 'unclassified']
            return non_unc[0] if non_unc else 'unclassified'
        df = (
            df.groupby(['flow_rate', 'actual_voltage'], as_index=False)
              .agg(classification=('classification', image_agg))
        )
    else:
        # Most common label (mode); ties broken by first occurrence
        df = (
            df.groupby(['flow_rate', 'actual_voltage'], as_index=False)
              .agg(classification=('classification', lambda x: x.mode().iloc[0]))
        )

    fig, ax = plt.subplots(figsize=(16, 8))

    def color_area(label, alpha_val):
        points = df[df['classification'] == label][['flow_rate', 'actual_voltage']].dropna().values
        if len(points) <= 5:
            return

        x_log = np.log10(points[:, 0])
        y_raw = points[:, 1]

        x_min, x_max = x_log.min(), x_log.max()
        y_min, y_max = y_raw.min(), y_raw.max()
        if x_max == x_min or y_max == y_min:
            return

        x_norm = (x_log - x_min) / (x_max - x_min)
        y_norm = (y_raw - y_min) / (y_max - y_min)
        points_norm = np.column_stack((x_norm, y_norm))

        try:
            hull = alphashape.alphashape(points_norm, alpha_val)
            geoms = (
                [hull]           if hull.geom_type == 'Polygon'      else
                list(hull.geoms) if hull.geom_type == 'MultiPolygon' else
                []
            )
            for poly in geoms:
                coords = np.array(poly.exterior.coords)
                real_x = 10 ** (coords[:, 0] * (x_max - x_min) + x_min)
                real_y = coords[:, 1] * (y_max - y_min) + y_min
                ax.add_patch(MplPolygon(
                    np.column_stack((real_x, real_y)),
                    alpha=0.15,
                    color=class_palette.get(label, '#000000'),
                    zorder=0, lw=0,
                ))
        except Exception as e:
            print(f"  Hull error for '{label}' in '{source_col}': {e}")

    sns.scatterplot(
        data=df, x='flow_rate', y='actual_voltage', hue='classification',
        palette=class_palette, alpha=0.8, edgecolor='none', s=70, zorder=2, ax=ax,
    )

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}' if x >= 1 else f'{x}'))
    ax.set_xlabel('Flow Rate ($\\mu L/min$)')
    ax.set_ylabel('Voltage ($V$)')
    ax.set_title(
        f'{SOLUTION_NORM}\nElectrospray Stability Map\n'
        f'(Classification Method: {source_col})'
    )
    ax.legend(title='Classification', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    os.makedirs('data/plots', exist_ok=True)
    filename = f'data/plots/stability_map_{SOLUTION_NORM}_{source_col}.png'
    plt.savefig(filename, dpi=300)
    print(f"  Saved: {filename}")
    plt.close()

def create_comparison_plot(truth_col, pred_col, data):
    df = data.copy()
    df['truth'] = df[truth_col].apply(clean_label)
    df['pred'] = df[pred_col].apply(clean_label)

    # --- Aggregate duplicate (flow_rate, actual_voltage) points ---
    def image_agg(labels):
        non_unc = [l for l in labels if l != 'unclassified']
        return non_unc[0] if non_unc else 'unclassified'
        
    df = (
        df.groupby(['flow_rate', 'actual_voltage'], as_index=False)
          .agg(truth=('truth', image_agg), pred=('pred', lambda x: x.mode().iloc[0]))
    )

    df['match'] = df['truth'] == df['pred']
    df['plot_color_label'] = df.apply(lambda row: row['truth'] if row['match'] else 'Incorrect', axis=1)

    fig, ax = plt.subplots(figsize=(16, 8))

    custom_palette = class_palette.copy()
    custom_palette['Incorrect'] = '#000000'

    sns.scatterplot(
        data=df, x='flow_rate', y='actual_voltage', hue='plot_color_label',
        palette=custom_palette, alpha=0.8, edgecolor='none', s=70, zorder=2, ax=ax,
    )

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}' if x >= 1 else f'{x}'))
    ax.set_xlabel('Flow Rate ($\\mu L/min$)')
    ax.set_ylabel('Voltage ($V$)')
    ax.set_title(
        f'{SOLUTION_NORM}\nComparison: {pred_col} vs {truth_col} (Truth)'
    )
    
    ax.legend(title='Classification (Match/Incorrect)', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Confusion matrix
    valid_mask = ~df['truth'].isin(['unclassified', 'EXCLUDE', 'none'])
    df_eval = df[valid_mask]
    
    if len(df_eval) > 0:
        accuracy = (df_eval['match'].sum() / len(df_eval)) * 100
        
        all_labels = sorted(list(set(df_eval['truth'].unique()).union(set(df_eval['pred'].unique()))))
        truth_cat = pd.Categorical(df_eval['truth'], categories=all_labels)
        pred_cat = pd.Categorical(df_eval['pred'], categories=all_labels)
        
        cm = pd.crosstab(truth_cat, pred_cat, normalize='index', dropna=False) * 100
        
        # Inset axes below the legend
        ax_cm = ax.inset_axes([1.02, 0.0, 0.35, 0.45])
        
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', ax=ax_cm, cbar=False, 
                    annot_kws={"size": 8}, vmin=0, vmax=100)
        ax_cm.set_title(f"Confusion Matrix (%)\nOverall Acc: {accuracy:.1f}%", fontsize=10)
        ax_cm.set_ylabel('Truth', fontsize=9)
        ax_cm.set_xlabel('Prediction', fontsize=9)
        ax_cm.set_xticklabels(ax_cm.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation=0, fontsize=8)
    else:
        ax_cm = ax.inset_axes([1.02, 0.0, 0.35, 0.45])
        ax_cm.axis('off')
        ax_cm.text(0.5, 0.5, "No valid data", ha='center', va='center')

    plt.tight_layout()
    os.makedirs('data/plots', exist_ok=True)
    filename = f'data/plots/comparison_{SOLUTION_NORM}_{truth_col}_vs_{pred_col}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
for source in available_sources:
    print(f"Generating plot for '{SOLUTION_NORM}': {source}")
    create_stability_plot(source, df_raw)

if COMPARISON_MAP:
    if 'image_classification' in df_raw.columns and 'rf_spray_mode' in df_raw.columns:
        print(f"Generating comparison plot for '{SOLUTION_NORM}': image_classification vs rf_spray_mode")
        create_comparison_plot('image_classification', 'rf_spray_mode', df_raw)
    if 'image_classification' in df_raw.columns and 'classical_classification' in df_raw.columns:
        print(f"Generating comparison plot for '{SOLUTION_NORM}': image_classification vs classical_classification")
        create_comparison_plot('image_classification', 'classical_classification', df_raw)
    if 'image_classification' in df_raw.columns and 'xgb_spray_mode' in df_raw.columns:
        print(f"Generating comparison plot for '{SOLUTION_NORM}': image_classification vs xgb_spray_mode")
        create_comparison_plot('image_classification', 'xgb_spray_mode', df_raw)

print("\nDone. All stability maps have been generated.")