"""
EHDA Spray Mode Classifier
===========================
Trains a Random Forest and XGBoost on the extracted + normalized features,
evaluates with cross-validation and a held-out test set, plots feature
importance, and saves everything needed for live inference.

USAGE:
    # From your existing pipeline:
    from feature_extraction import process_json_file, process_multiple_files
    from ehda_normalization import prepare_training_data
    from ehda_classifier import train, load_classifier, predict

    df  = process_multiple_files("*.json", folder="dataset/")
    df_norm, X, labels, feature_names, normalizer = prepare_training_data(df)
    results = train(X, labels, feature_names)

    # Live inference:
    clf = load_classifier("models/")
    pred, proba = predict(clf, x_new_normalized)

REQUIREMENTS:
    pip install scikit-learn xgboost matplotlib seaborn joblib
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
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not installed — skipping XGB model. Run: pip install xgboost")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path("models")
PLOT_DIR   = Path("plots")
RANDOM_STATE = 42
TEST_SIZE    = 0.2    # 20% held out for final evaluation
CV_FOLDS     = 5      # stratified k-fold cross-validation


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def _build_random_forest(n_classes: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,       # more trees = more stable, diminishing returns after ~300
        max_depth=None,         # grow full trees — RF handles overfitting via bagging
        min_samples_leaf=2,     # prevents single-sample leaves
        max_features="sqrt",    # sqrt(n_features) per split — standard for classification
        class_weight="balanced",# handles imbalanced spray mode counts automatically
        n_jobs=-1,              # use all CPU cores
        random_state=RANDOM_STATE,
    )


def _build_xgboost(n_classes: int) -> "XGBClassifier":
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,          # row subsampling per tree (like RF bagging)
        colsample_bytree=0.8,   # feature subsampling per tree
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def _cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                           model_name: str) -> dict:
    """Run stratified k-fold CV and report per-fold metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1_macro", "f1_weighted"]

    print(f"\n  Running {CV_FOLDS}-fold cross-validation on {model_name}...")
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                 return_train_score=True, n_jobs=-1)

    print(f"  {'Metric':<25} {'Train':>10} {'Val':>10}")
    print(f"  {'-'*47}")
    for metric in scoring:
        train_m = cv_results[f"train_{metric}"]
        val_m   = cv_results[f"test_{metric}"]
        print(f"  {metric:<25} {train_m.mean():.4f}±{train_m.std():.4f}"
              f"  {val_m.mean():.4f}±{val_m.std():.4f}")

    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def _plot_feature_importance(model, feature_names: list,
                              model_name: str, top_n: int = 25) -> pd.DataFrame:
    """Extract and plot the top N most important features."""
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Assign feature group for color coding
    def _group(name):
        if name in ("target_voltage", "actual_voltage", "voltage_error",
                    "flow_rate", "current_PS"):
            return "Metadata"
        if name.startswith(("band_", "spectral", "dominant", "mean_freq",
                             "median_freq", "total_power")):
            return "Frequency"
        if name.startswith("wt_"):
            return "Wavelet"
        return "Time-domain"

    df_imp["group"] = df_imp["feature"].apply(_group)

    palette = {"Time-domain": "#4C9BE8", "Frequency": "#E8834C",
               "Wavelet": "#6DBE6D", "Metadata": "#B06DBE"}

    fig, ax = plt.subplots(figsize=(10, 8))
    top = df_imp.head(top_n)
    colors = [palette[g] for g in top["group"]]
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title(f"{model_name} — Top {top_n} Features")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in palette.items()
                       if g in top["group"].values]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"feature_importance_{model_name.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return df_imp


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, class_names: list,
                            model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    # Normalized version (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalized (Recall)"],
        ["d", ".2f"]
    ):
        disp = ConfusionMatrixDisplay(data, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=fmt)
        ax.set_title(f"{model_name} — {title}")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"confusion_matrix_{model_name.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-CLASS METRICS PLOT
# ─────────────────────────────────────────────────────────────────────────────
def _plot_per_class_metrics(y_true, y_pred, class_names: list,
                             model_name: str) -> None:
    report = classification_report(y_true, y_pred,
                                    target_names=class_names, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    df_r = pd.DataFrame({m: [report[c][m] for c in class_names]
                         for m in metrics}, index=class_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(class_names))
    width = 0.25
    colors = ["#4C9BE8", "#E8834C", "#6DBE6D"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df_r[metric], width, label=metric.capitalize(),
               color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Per-Class Metrics (Test Set)")
    ax.legend()
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"per_class_{model_name.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def _train_one(model, model_name: str,
               X_train, y_train, X_test, y_test,
               feature_names: list, class_names: list,
               label_encoder: LabelEncoder) -> dict:

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    # Cross-validation on training set
    cv_results = _cross_validate_model(model, X_train, y_train, model_name)

    # Final fit on full training set
    print(f"\n  Fitting on full training set ({len(X_train)} samples)...")
    model.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    print(f"\n  Test set results ({len(X_test)} samples):")
    print(classification_report(y_test_labels, y_pred_labels,
                                  target_names=class_names))

    # Plots
    _plot_confusion_matrix(y_test_labels, y_pred_labels,
                            class_names, model_name)
    _plot_per_class_metrics(y_test_labels, y_pred_labels,
                             class_names, model_name)
    df_imp = _plot_feature_importance(model, feature_names, model_name)

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"{model_name.lower().replace(' ','_')}.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved model: {model_path}")

    return {
        "model":        model,
        "model_name":   model_name,
        "cv_results":   cv_results,
        "importances":  df_imp,
        "cv_val_acc":   cv_results["test_accuracy"].mean(),
        "cv_val_f1":    cv_results["test_f1_macro"].mean(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE MODELS PLOT
# ─────────────────────────────────────────────────────────────────────────────
def _plot_model_comparison(all_results: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics  = ["cv_val_acc", "cv_val_f1"]
    titles   = ["CV Validation Accuracy", "CV Validation F1 (Macro)"]
    colors   = ["#4C9BE8", "#E8834C", "#6DBE6D", "#B06DBE"]

    for ax, metric, title in zip(axes, metrics, titles):
        names  = [r["model_name"] for r in all_results]
        values = [r[metric] for r in all_results]
        bars = ax.bar(names, values,
                      color=colors[:len(names)], alpha=0.85, width=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = PLOT_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train(X: np.ndarray, labels: np.ndarray,
          feature_names: list) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    X             : feature matrix from prepare_training_data()
    labels        : string label array  e.g. ["dripping", "cone_jet", ...]
    feature_names : list of feature column names (same order as X columns)

    Returns
    -------
    dict with keys: "random_forest", "xgboost" (if available),
                    "best_model", "label_encoder", "class_names"
    """
    # Encode string labels to integers (required by XGBoost, consistent for RF)
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    class_names = list(le.classes_)
    n_classes   = len(class_names)

    print(f"\n{'='*60}")
    print(f"  EHDA Classifier Training")
    print(f"{'='*60}")
    print(f"  Samples:      {len(X)}")
    print(f"  Features:     {len(feature_names)}")
    print(f"  Classes:      {class_names}")
    print(f"  Train/Test:   {int(len(X)*(1-TEST_SIZE))} / {int(len(X)*TEST_SIZE)}")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u:<25} {c:>4} samples  ({100*c/len(labels):.1f}%)")

    # Stratified train/test split — preserves class ratios in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    all_results = []

    # ── Random Forest ────────────────────────────────────────────────────────
    rf_result = _train_one(
        model        = _build_random_forest(n_classes),
        model_name   = "Random Forest",
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        label_encoder=le,
    )
    all_results.append(rf_result)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        xgb_result = _train_one(
            model        = _build_xgboost(n_classes),
            model_name   = "XGBoost",
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            feature_names=feature_names,
            class_names=class_names,
            label_encoder=le,
        )
        all_results.append(xgb_result)

    # ── Compare & pick best ──────────────────────────────────────────────────
    _plot_model_comparison(all_results)
    best = max(all_results, key=lambda r: r["cv_val_f1"])
    print(f"\n  Best model by CV F1: {best['model_name']}  "
          f"(F1={best['cv_val_f1']:.4f})")

    # Save label encoder and metadata — needed for inference
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(le,           MODEL_DIR / "label_encoder.pkl")
    joblib.dump(class_names,  MODEL_DIR / "class_names.pkl")
    joblib.dump(feature_names,MODEL_DIR / "feature_names.pkl")
    print(f"  Saved label encoder and metadata to {MODEL_DIR}/")

    # Print top 10 features from best model
    print(f"\n  Top 10 most important features ({best['model_name']}):")
    print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>10}")
    print(f"  {'-'*58}")
    for i, row in best["importances"].head(10).iterrows():
        print(f"  {i+1:<6} {row['feature']:<40} {row['importance']:>10.4f}")

    return {
        "random_forest":  rf_result,
        "xgboost":        xgb_result if XGBOOST_AVAILABLE else None,
        "best_model":     best["model"],
        "best_name":      best["model_name"],
        "label_encoder":  le,
        "class_names":    class_names,
        "all_results":    all_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
class EHDAClassifier:
    """
    Lightweight wrapper for live inference.
    Loads the saved model + label encoder from disk.

    Usage:
        clf = EHDAClassifier.load("models/", model_name="random_forest")
        pred, proba = clf.predict(x_normalized)
    """
    def __init__(self, model, label_encoder, class_names, feature_names):
        self.model         = model
        self.label_encoder = label_encoder
        self.class_names   = class_names
        self.feature_names = feature_names

    @classmethod
    def load(cls, folder: str = "models",
             model_name: str = "random_forest") -> "EHDAClassifier":
        folder = Path(folder)
        model         = joblib.load(folder / f"{model_name}.pkl")
        label_encoder = joblib.load(folder / "label_encoder.pkl")
        class_names   = joblib.load(folder / "class_names.pkl")
        feature_names = joblib.load(folder / "feature_names.pkl")
        print(f"✓ Loaded {model_name} from {folder}/")
        return cls(model, label_encoder, class_names, feature_names)

    def predict(self, x: np.ndarray) -> tuple:
        """
        Predict spray mode for a single normalized feature vector.

        Parameters
        ----------
        x : 1D numpy array from prepare_inference_sample()

        Returns
        -------
        prediction : str   e.g. "cone_jet"
        probabilities : dict  e.g. {"cone_jet": 0.91, "dripping": 0.07, ...}
        """
        x2d   = x.reshape(1, -1)
        y_enc = self.model.predict(x2d)[0]
        proba = self.model.predict_proba(x2d)[0]

        prediction    = self.label_encoder.inverse_transform([y_enc])[0]
        probabilities = dict(zip(self.class_names, proba.tolist()))

        return prediction, probabilities


def load_classifier(folder: str = "models",
                    model_name: str = "random_forest") -> EHDAClassifier:
    """Convenience function — same as EHDAClassifier.load()."""
    return EHDAClassifier.load(folder, model_name)


def predict(clf: EHDAClassifier, x: np.ndarray) -> tuple:
    """Convenience function — same as clf.predict(x)."""
    return clf.predict(x)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from feature_extraction import process_multiple_files
    from ehda_normalization import prepare_training_data

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Loading data from: {folder}")

    df = process_multiple_files("*.json", folder=folder)
    df_norm, X, labels, feature_names, normalizer = prepare_training_data(df, drop_metadata=True, exclude_label="EXCLUDE")

    results = train(X, labels, feature_names)

    print(f"\n{'='*60}")
    print(f"  DONE — files saved to:")
    print(f"    models/   — trained model + label encoder")
    print(f"    plots/    — feature importance, confusion matrices")
    print(f"{'='*60}")