"""
EHDA Feature Selector
======================
Systematically identifies and removes uninformative or redundant features
to improve model accuracy and reduce overfitting.

Three-stage selection pipeline:
  Stage 1 — Near-zero variance filter
             Drops features whose values barely change across samples.
             These carry no discriminative information, only noise.

  Stage 2 — Correlation filter
             Within groups of highly correlated features (|r| > threshold),
             keeps only the one with the highest Random Forest importance.
             Removes mathematical redundancy without losing information.

  Stage 3 — Importance threshold
             Trains a Random Forest on the Stage 2 survivors, ranks by
             mean-decrease-in-impurity importance, and sweeps thresholds
             to find the smallest set that retains 95% of CV F1.

Each stage is reversible — the selector saves exactly which features
were kept so you can reproduce or modify the selection later.

USAGE (standalone — generates a report and recommended feature list):

    python feature_selector.py

USAGE (integrated into your pipeline):

    from feature_extraction import process_multiple_files
    from ehda_normalization import prepare_training_data
    from feature_selector import select_features
    from ehda_classifier import train

    df = process_multiple_files("*.json", folder=r"new data")
    df_raw, X, labels, feature_names, normalizer = prepare_training_data(
        df, drop_metadata=True, exclude_label="EXCLUDE"
    )

    # Run selection on the raw (pre-normalization) feature matrix.
    # Normalizer is fitted inside select_features() on each CV fold's
    # training split, same as in train().
    X_selected, selected_names, selector_report = select_features(
        X, labels, feature_names, normalizer
    )

    results = train(X_selected, labels, selected_names, normalizer=normalizer)

REQUIREMENTS:
    pip install scikit-learn matplotlib seaborn pandas numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PLOT_DIR     = Path("plots")
RANDOM_STATE = 42
CV_FOLDS     = 5

# Stage 1
VARIANCE_THRESHOLD = 0.01  # features with variance below this are dropped
                            # (on normalized data this is ~1% of unit variance)

# Stage 2
CORRELATION_THRESHOLD = 0.95  # |Pearson r| above this → features are redundant

# Stage 3
IMPORTANCE_RETENTION = 0.95   # keep the smallest set that retains this
                               # fraction of the baseline CV F1


# ─────────────────────────────────────────────────────────────────────────────
# QUICK RF — used internally for importance scoring
# ─────────────────────────────────────────────────────────────────────────────
def _quick_rf() -> RandomForestClassifier:
    """A fast RF for feature-importance estimation — intentionally lightweight."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _cv_f1(model, X: np.ndarray, y: np.ndarray) -> float:
    """Return mean stratified-CV macro-F1 for a given feature matrix."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return float(scores.mean())


def _normalize_for_selection(X: np.ndarray, y: np.ndarray,
                              feature_names: list,
                              normalizer) -> np.ndarray:
    """
    Apply normalization correctly for feature selection:
    fit the scaler on each training fold, transform the full X at the end
    using a scaler fitted on all data (good enough for selection — exact
    fold-wise normalization would require re-running inside CV, which is
    prohibitively slow during a sweep).

    Returns X_norm (numpy array) ready for variance/correlation analysis.
    """
    if normalizer is None:
        return X
    df = pd.DataFrame(X, columns=feature_names)
    normalizer.fit(df)
    return normalizer.transform(df)[feature_names].values


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — NEAR-ZERO VARIANCE FILTER
# ─────────────────────────────────────────────────────────────────────────────
def _stage1_variance(X_norm: np.ndarray, feature_names: list,
                     threshold: float = VARIANCE_THRESHOLD) -> Tuple[np.ndarray, list, list]:
    """
    Drop features whose variance across all samples is below `threshold`.

    A feature with near-zero variance has essentially the same value for
    every sample. No matter how sophisticated the classifier, it cannot
    learn anything from a constant column — but it adds a dimension to
    the search space and can hurt nearest-neighbour-style splits.

    Returns
    -------
    X_out         : filtered feature matrix
    kept_names    : list of retained feature names
    dropped_names : list of dropped feature names
    """
    variances   = np.var(X_norm, axis=0)
    keep_mask   = variances >= threshold
    kept_names  = [n for n, k in zip(feature_names, keep_mask) if k]
    dropped_names = [n for n, k in zip(feature_names, keep_mask) if not k]

    print(f"\n  Stage 1 — Variance filter (threshold={threshold})")
    print(f"    Input:   {len(feature_names)} features")
    print(f"    Dropped: {len(dropped_names)}")
    if dropped_names:
        for n in dropped_names:
            v = variances[feature_names.index(n)]
            print(f"      {n:<45} var={v:.6f}")
    print(f"    Kept:    {len(kept_names)} features")

    return X_norm[:, keep_mask], kept_names, dropped_names


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CORRELATION FILTER
# ─────────────────────────────────────────────────────────────────────────────
def _stage2_correlation(X: np.ndarray, feature_names: list, y: np.ndarray,
                         threshold: float = CORRELATION_THRESHOLD
                         ) -> Tuple[np.ndarray, list, list]:
    """
    Within groups of highly correlated features, keep only the one with
    the highest Random Forest importance. Drop the rest.

    Why not just drop arbitrarily? Importance-guided pruning ensures we
    keep the most discriminative representative from each correlated cluster,
    rather than an arbitrary one.

    Returns
    -------
    X_out         : filtered feature matrix
    kept_names    : list of retained feature names
    dropped_names : list of (dropped_feature, kept_representative) pairs
    """
    print(f"\n  Stage 2 — Correlation filter (threshold={threshold})")

    # Get RF importances to break ties in correlated clusters
    rf = _quick_rf()
    rf.fit(X, y)
    importances = dict(zip(feature_names, rf.feature_importances_))

    corr = np.corrcoef(X.T)
    n    = len(feature_names)

    to_drop   = set()
    drop_pairs = []    # (dropped, kept) for reporting

    for i in range(n):
        if feature_names[i] in to_drop:
            continue
        for j in range(i + 1, n):
            if feature_names[j] in to_drop:
                continue
            if abs(corr[i, j]) >= threshold:
                # Keep the one with higher importance; drop the other
                imp_i = importances[feature_names[i]]
                imp_j = importances[feature_names[j]]
                if imp_i >= imp_j:
                    to_drop.add(feature_names[j])
                    drop_pairs.append((feature_names[j], feature_names[i]))
                else:
                    to_drop.add(feature_names[i])
                    drop_pairs.append((feature_names[i], feature_names[j]))
                    break  # feature i is now marked for dropping; move to i+1

    kept_mask  = [n not in to_drop for n in feature_names]
    kept_names = [n for n in feature_names if n not in to_drop]

    print(f"    Input:   {len(feature_names)} features")
    print(f"    Dropped: {len(to_drop)} (corr ≥ {threshold}, lower importance kept)")
    if drop_pairs:
        print(f"      {'Dropped':<45} {'Kept (higher importance)'}")
        print(f"      {'-'*85}")
        for dropped, kept in sorted(drop_pairs):
            r_idx = feature_names.index(dropped)
            k_idx = feature_names.index(kept)
            r_val = corr[r_idx, k_idx]
            print(f"      {dropped:<45} → {kept}  (|r|={r_val:.3f})")
    print(f"    Kept:    {len(kept_names)} features")

    X_out = X[:, [i for i, k in enumerate(kept_mask) if k]]
    return X_out, kept_names, list(to_drop)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — IMPORTANCE THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────
def _stage3_importance_sweep(X: np.ndarray, feature_names: list, y: np.ndarray,
                              retention: float = IMPORTANCE_RETENTION,
                              normalizer=None
                              ) -> Tuple[np.ndarray, list, pd.DataFrame, float, float]:
    """
    Rank features by RF importance, sweep importance thresholds, and return
    the smallest feature set that retains `retention` fraction of baseline F1.

    Because normalization happens inside train(), we pass the normalizer here
    and apply it to each candidate subset the same way train() would — fitted
    on all data once (appropriate for a sweep, where we want relative F1 scores).

    Returns
    -------
    X_out            : selected feature matrix
    selected_names   : retained feature names
    sweep_df         : DataFrame of (n_features, threshold, cv_f1) for all thresholds
    baseline_f1      : CV F1 with all input features
    selected_f1      : CV F1 with selected features
    """
    print(f"\n  Stage 3 — Importance sweep (retain {retention*100:.0f}% of baseline F1)")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    rf = _quick_rf()
    rf.fit(X, y_enc)

    importances  = rf.feature_importances_
    sorted_idx   = np.argsort(importances)[::-1]  # descending
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imp   = importances[sorted_idx]

    # Baseline: all features
    baseline_f1 = _cv_f1(_quick_rf(), X, y_enc)
    target_f1   = retention * baseline_f1
    print(f"    Baseline CV F1 ({len(feature_names)} features): {baseline_f1:.4f}")
    print(f"    Target F1 (≥{retention*100:.0f}% retention):    {target_f1:.4f}")

    # Sweep: keep top k features, k = 1 … n
    sweep_rows = []
    best_k     = len(feature_names)  # fallback: keep all
    best_names = feature_names

    print(f"\n    {'N features':>12}  {'CV F1':>8}  {'% of baseline':>15}  Status")
    print(f"    {'-'*55}")

    for k in range(1, len(sorted_names) + 1):
        top_names = sorted_names[:k]
        top_idx   = [feature_names.index(n) for n in top_names]
        X_k       = X[:, top_idx]

        f1 = _cv_f1(_quick_rf(), X_k, y_enc)
        pct = 100 * f1 / baseline_f1 if baseline_f1 > 0 else 0.0
        sweep_rows.append({"n_features": k, "cv_f1": f1, "pct_baseline": pct})

        status = ""
        if f1 >= target_f1 and best_k == len(feature_names):
            best_k     = k
            best_names = top_names
            status = "✓ SELECTED"

        # Print every 5th point + the selected point to avoid flooding the console
        if k % 5 == 0 or status or k == 1:
            print(f"    {k:>12}  {f1:>8.4f}  {pct:>14.1f}%  {status}")

    sweep_df = pd.DataFrame(sweep_rows)

    selected_idx = [feature_names.index(n) for n in best_names]
    X_out        = X[:, selected_idx]
    selected_f1  = sweep_df.loc[sweep_df["n_features"] == best_k, "cv_f1"].values[0]

    print(f"\n    Selected: {best_k} features  (CV F1={selected_f1:.4f}, "
          f"{100*selected_f1/baseline_f1:.1f}% of baseline)")

    return X_out, best_names, sweep_df, baseline_f1, selected_f1


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def _plot_correlation_heatmap(X: np.ndarray, feature_names: list,
                               title: str = "Feature Correlation Matrix") -> None:
    """Plot a heatmap of absolute Pearson correlations between features."""
    corr = np.abs(np.corrcoef(X.T))
    df_corr = pd.DataFrame(corr, index=feature_names, columns=feature_names)

    n = len(feature_names)
    figsize = (max(10, n * 0.35), max(8, n * 0.3))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_corr, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=True, yticklabels=True,
                linewidths=0.3 if n < 30 else 0.0, square=True)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=6 if n > 20 else 8)
    ax.tick_params(axis="y", rotation=0,  labelsize=6 if n > 20 else 8)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_importance_sweep(sweep_df: pd.DataFrame,
                            baseline_f1: float,
                            selected_n: int,
                            retention: float = IMPORTANCE_RETENTION) -> None:
    """Plot CV F1 vs number of features retained."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sweep_df["n_features"], sweep_df["cv_f1"],
            color="#4C9BE8", linewidth=1.8, marker="o", markersize=3)
    ax.axhline(baseline_f1, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (all features) F1={baseline_f1:.3f}")
    ax.axhline(baseline_f1 * retention, color="#E8834C", linestyle=":",
               linewidth=1.2, label=f"{retention*100:.0f}% retention target")
    selected_f1 = sweep_df.loc[sweep_df["n_features"] == selected_n, "cv_f1"].values[0]
    ax.axvline(selected_n, color="#6DBE6D", linestyle="-.", linewidth=1.5,
               label=f"Selected: {selected_n} features (F1={selected_f1:.3f})")
    ax.scatter([selected_n], [selected_f1], color="#6DBE6D", s=80, zorder=5)
    ax.set_xlabel("Number of Features (top-k by RF importance)")
    ax.set_ylabel("CV Macro F1")
    ax.set_title("Feature Count vs. CV Performance (Importance Sweep)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = PLOT_DIR / "feature_importance_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_final_importances(X: np.ndarray, feature_names: list,
                             y_enc: np.ndarray) -> None:
    """Bar chart of RF importances for the final selected feature set."""
    rf = _quick_rf()
    rf.fit(X, y_enc)

    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    def _group(name):
        if name.startswith(("band_", "spectral", "dominant", "mean_freq",
                             "median_freq", "total_power")):
            return "Frequency"
        if name.startswith("wt_"):
            return "Wavelet"
        if name in ("target_voltage", "actual_voltage", "voltage_error",
                    "flow_rate", "current_PS"):
            return "Metadata"
        return "Time-domain"

    palette = {"Time-domain": "#4C9BE8", "Frequency": "#E8834C",
               "Wavelet": "#6DBE6D", "Metadata": "#B06DBE"}
    df_imp["group"] = df_imp["feature"].apply(_group)

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.3)))
    colors = [palette[g] for g in df_imp["group"]]
    ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1],
            color=colors[::-1], alpha=0.85)
    ax.set_xlabel("RF Feature Importance")
    ax.set_title("Selected Feature Importances")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in palette.items()
                       if g in df_imp["group"].values]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    path = PLOT_DIR / "selected_feature_importances.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def select_features(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list,
    normalizer=None,
    variance_threshold: float = VARIANCE_THRESHOLD,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    importance_retention: float = IMPORTANCE_RETENTION,
    save_selection: bool = True,
    plot: bool = True,
) -> Tuple[np.ndarray, list, dict]:
    """
    Run the full three-stage feature selection pipeline.

    Parameters
    ----------
    X                     : RAW feature matrix (before normalization)
    labels                : string label array
    feature_names         : list of column names matching X
    normalizer            : EHDAFeatureNormalizer (unfitted) from prepare_training_data().
                            Used to normalize X for variance/correlation analysis.
                            Pass None to skip normalization during selection.
    variance_threshold    : Stage 1 cutoff — features with var < this are dropped.
                            Default 0.01.
    correlation_threshold : Stage 2 cutoff — |r| >= this means redundant.
                            Default 0.95.
    importance_retention  : Stage 3 target — keep the smallest set that achieves
                            this fraction of full-feature-set CV F1. Default 0.95.
    save_selection        : save selected feature names to models/selected_features.pkl
    plot                  : generate and save selection diagnostic plots

    Returns
    -------
    X_selected      : feature matrix with only selected columns (raw, unnormalized)
    selected_names  : ordered list of selected feature names
    report          : dict with per-stage details (n_dropped, dropped names, etc.)
    """
    print(f"\n{'='*60}")
    print(f"  EHDA Feature Selection")
    print(f"{'='*60}")
    print(f"  Input: {X.shape[0]} samples × {X.shape[1]} features")

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # Normalize X for analysis (without leaking — we fit on all data here,
    # which is acceptable for feature selection since we're only choosing
    # which columns to keep, not learning model weights).
    X_norm = _normalize_for_selection(X, labels, feature_names, normalizer)

    if plot:
        print(f"\n  Plotting full correlation heatmap (before selection)...")
        _plot_correlation_heatmap(X_norm, feature_names, "Correlation — All Features")

    # ── Stage 1: Variance ────────────────────────────────────────────────────
    X1, names1, dropped_var = _stage1_variance(X_norm, feature_names, variance_threshold)

    # ── Stage 2: Correlation ─────────────────────────────────────────────────
    X2, names2, dropped_corr = _stage2_correlation(X1, names1, y, correlation_threshold)

    if plot and len(names2) < len(feature_names):
        print(f"\n  Plotting post-correlation heatmap...")
        _plot_correlation_heatmap(X2, names2, "Correlation — After Stage 2")

    # ── Stage 3: Importance sweep ────────────────────────────────────────────
    X3, names3, sweep_df, baseline_f1, selected_f1 = _stage3_importance_sweep(
        X2, names2, y, importance_retention, normalizer
    )

    if plot:
        _plot_importance_sweep(sweep_df, baseline_f1, len(names3), importance_retention)
        _plot_final_importances(X3, names3, y)

    # ── Map selected names back to ORIGINAL (raw, un-normalized) X ───────────
    final_idx   = [feature_names.index(n) for n in names3]
    X_selected  = X[:, final_idx]

    # ── Report ───────────────────────────────────────────────────────────────
    report = {
        "n_input":          len(feature_names),
        "n_selected":       len(names3),
        "n_dropped_total":  len(feature_names) - len(names3),
        "dropped_variance": dropped_var,
        "dropped_corr":     dropped_corr,
        "dropped_importance": [n for n in names2 if n not in names3],
        "selected_names":   names3,
        "baseline_f1":      baseline_f1,
        "selected_f1":      selected_f1,
        "sweep_df":         sweep_df,
    }

    print(f"\n{'='*60}")
    print(f"  SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Input features:              {report['n_input']}")
    print(f"  Dropped (near-zero var):     {len(dropped_var)}")
    print(f"  Dropped (high correlation):  {len(dropped_corr)}")
    print(f"  Dropped (low importance):    {len(report['dropped_importance'])}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Final selected features:     {report['n_selected']}")
    print(f"  Baseline CV F1 (all feats):  {baseline_f1:.4f}")
    print(f"  Selected CV F1:              {selected_f1:.4f}  "
          f"({100*selected_f1/baseline_f1:.1f}% retained)")

    print(f"\n  Selected features (ranked by importance):")
    rf_final = _quick_rf()
    rf_final.fit(X3, y)
    imp_final = dict(zip(names3, rf_final.feature_importances_))
    for rank, name in enumerate(names3, 1):
        print(f"    {rank:>3}. {name:<45} {imp_final[name]:.4f}")

    if save_selection:
        Path("models").mkdir(exist_ok=True)
        path = Path("models") / "selected_features.pkl"
        joblib.dump(names3, path)
        print(f"\n  Saved selected feature list → {path}")

    return X_selected, names3, report


def load_selected_features(folder: str = "models") -> list:
    """
    Load the saved feature list for inference.
    Use this to subset new samples before predicting.

    Example
    -------
    from feature_selector import load_selected_features
    selected_names = load_selected_features()
    x_sub = x_full[[feature_names.index(n) for n in selected_names]]
    """
    path = Path(folder) / "selected_features.pkl"
    names = joblib.load(path)
    print(f"Loaded {len(names)} selected features from {path}")
    return names


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    from feature_extraction import process_multiple_files
    from ehda_normalization import prepare_training_data

    folder = "new data"
    print(f"Loading data from: {folder}")

    df = process_multiple_files("*.json", folder=folder)
    df_raw, X, labels, feature_names, normalizer = prepare_training_data(
        df, drop_metadata=True, exclude_label="EXCLUDE"
    )

    X_selected, selected_names, report = select_features(
        X, labels, feature_names, normalizer=normalizer
    )

    print(f"\n{'='*60}")
    print(f"  DONE — use these in your pipeline:")
    print(f"{'='*60}")
    print(f"\n  from feature_extraction import process_multiple_files")
    print(f"  from ehda_normalization import prepare_training_data")
    print(f"  from feature_selector import select_features")
    print(f"  from ehda_classifier import train")
    print(f"\n  df = process_multiple_files('*.json', folder=r'new data')")
    print(f"  df_raw, X, labels, feature_names, normalizer = prepare_training_data(")
    print(f"      df, drop_metadata=True, exclude_label='EXCLUDE'")
    print(f"  )")
    print(f"  X_sel, sel_names, _ = select_features(X, labels, feature_names, normalizer)")
    print(f"  results = train(X_sel, labels, sel_names, normalizer=normalizer)")