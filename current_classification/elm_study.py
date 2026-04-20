"""
EHDA Feature Group Ablation Study for ELM
==========================================
Systematically tests different feature subsets with ELM to identify which
feature groups (time-domain, frequency-domain, wavelet, metadata) contribute
most to EHDA spray mode classification.

Includes label filtering to exclude unwanted classes like "unconclusive" or
"undefined", and warns about classes with very few samples.

USAGE:
    python elm_feature_ablation.py [data_folder]

    # Or integrate:
    from elm_feature_ablation import run_ablation_study
    results_df, best_subset = run_ablation_study(
        df, 
        exclude_labels=['EXCLUDE', 'unconclusive', 'undefined', 'N/A']
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
import joblib
import time
from itertools import combinations

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2
MIN_SAMPLES_PER_CLASS = 3  # Warn if any class has fewer than this after filtering
PLOT_DIR = Path("plots/elm_ablation")
MODEL_DIR = Path("models/elm_ablation")

PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ELM CLASSIFIER (Optimized version)
# ============================================================================
class ELMClassifier(BaseEstimator, ClassifierMixin):
    """Extreme Learning Machine with regularization."""
    
    def __init__(self, n_hidden=1000, activation='tanh', C=1e-3, random_state=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.C = C
        self.random_state = random_state
        self.lb_ = None
        self.w_ = None
        self.b_ = None
        self.beta_ = None
        
    def _activate(self, X):
        if self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'relu':
            return np.maximum(X, 0)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        else:
            return X

    def fit(self, X, y):
        self.lb_ = LabelBinarizer()
        Y = self.lb_.fit_transform(y)
        if Y.shape[1] == 1:
            Y = np.hstack((1 - Y, Y))
            
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        
        self.w_ = rng.normal(scale=0.1, size=(n_features, self.n_hidden))
        self.b_ = rng.normal(scale=0.1, size=self.n_hidden)
        
        H = self._activate(np.dot(X, self.w_) + self.b_)
        I = np.eye(self.n_hidden)
        self.beta_ = np.linalg.solve(H.T @ H + self.C * I, H.T @ Y)
        return self
        
    def predict_proba(self, X):
        H = self._activate(np.dot(X, self.w_) + self.b_)
        raw_preds = np.dot(H, self.beta_)
        max_rep = np.max(raw_preds, axis=1, keepdims=True)
        exp_preds = np.exp(raw_preds - max_rep)
        return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
    def predict(self, X):
        H = self._activate(np.dot(X, self.w_) + self.b_)
        raw_preds = np.dot(H, self.beta_)
        return self.lb_.classes_[np.argmax(raw_preds, axis=1)]


# ============================================================================
# FEATURE GROUP DEFINITIONS
# ============================================================================
def get_feature_groups(feature_names):
    """
    Categorize features into logical groups based on naming conventions.
    Returns a dict mapping group name -> list of feature names.
    """
    groups = {
        'time_domain': [],
        'frequency_domain': [],
        'wavelet': [],
        'metadata': [],
        'statistical_moments': [],  # kurtosis, skewness, crest factor, shape factor
    }
    
    for feat in feature_names:
        # Metadata / operating conditions
        if feat in ['target_voltage', 'actual_voltage', 'voltage_error', 
                    'flow_rate', 'current_PS']:
            groups['metadata'].append(feat)
            
        # Frequency domain features
        elif feat.startswith(('band_', 'spectral_', 'dominant_freq', 
                              'mean_freq', 'median_freq', 'total_power')):
            groups['frequency_domain'].append(feat)
            
        # Wavelet features
        elif feat.startswith('wt_'):
            groups['wavelet'].append(feat)
            
        # Statistical moments (dimensionless shape descriptors)
        elif feat in ['kurtosis', 'skewness', 'crest_factor', 'shape_factor',
                      'zero_crossing_rate']:
            groups['statistical_moments'].append(feat)
            
        # Everything else goes to time_domain
        else:
            groups['time_domain'].append(feat)
    
    return groups


def create_feature_subsets(feature_names):
    """
    Create various feature subsets for ablation testing.
    Returns list of (subset_name, feature_list) tuples.
    """
    groups = get_feature_groups(feature_names)
    
    subsets = []
    
    # Individual groups
    subsets.append(("Time Domain Only", groups['time_domain']))
    subsets.append(("Frequency Domain Only", groups['frequency_domain']))
    subsets.append(("Wavelet Only", groups['wavelet']))
    subsets.append(("Statistical Moments Only", groups['statistical_moments']))
    subsets.append(("Metadata Only", groups['metadata']))
    
    # Logical combinations
    subsets.append(("Time + Frequency", 
                    groups['time_domain'] + groups['frequency_domain']))
    subsets.append(("Time + Wavelet", 
                    groups['time_domain'] + groups['wavelet']))
    subsets.append(("Frequency + Wavelet", 
                    groups['frequency_domain'] + groups['wavelet']))
    subsets.append(("Time + Frequency + Wavelet", 
                    groups['time_domain'] + groups['frequency_domain'] + groups['wavelet']))
    subsets.append(("Shape Features Only", 
                    groups['statistical_moments'] + groups['time_domain']))
    
    # With and without metadata
    base_all = (groups['time_domain'] + groups['frequency_domain'] + 
                groups['wavelet'] + groups['statistical_moments'])
    subsets.append(("All Features (no metadata)", base_all))
    subsets.append(("All Features (with metadata)", base_all + groups['metadata']))
    
    # Filter out empty subsets
    subsets = [(name, feats) for name, feats in subsets if len(feats) > 0]
    
    return subsets, groups


# ============================================================================
# DATA FILTERING
# ============================================================================
def filter_labels(df, label_column='label', exclude_labels=None):
    """
    Filter dataframe to remove unwanted labels.
    
    Parameters
    ----------
    df : DataFrame
    label_column : str
    exclude_labels : list of str
        Labels to exclude (e.g., ['EXCLUDE', 'unconclusive', 'undefined', 'N/A'])
    
    Returns
    -------
    df_filtered : DataFrame
    dropped_counts : dict
    """
    if exclude_labels is None:
        exclude_labels = ['EXCLUDE', 'unconclusive', 'undefined', 'N/A']
    
    original_count = len(df)
    df_filtered = df.copy()
    
    # Also drop NaN labels
    df_filtered = df_filtered.dropna(subset=[label_column])
    
    # Count and drop excluded labels
    dropped_counts = {}
    for lbl in exclude_labels:
        mask = df_filtered[label_column] == lbl
        dropped_counts[lbl] = mask.sum()
        df_filtered = df_filtered[~mask]
    
    kept_count = len(df_filtered)
    total_dropped = original_count - kept_count
    
    print(f"\n  Label filtering:")
    print(f"    Original samples: {original_count}")
    for lbl, count in dropped_counts.items():
        if count > 0:
            print(f"    Dropped '{lbl}': {count}")
    print(f"    Kept samples: {kept_count}")
    
    return df_filtered, dropped_counts


def check_class_balance(labels, min_samples=MIN_SAMPLES_PER_CLASS):
    """Warn about classes with too few samples."""
    unique, counts = np.unique(labels, return_counts=True)
    low_count_classes = []
    for cls, cnt in zip(unique, counts):
        if cnt < min_samples:
            low_count_classes.append((cls, cnt))
    
    if low_count_classes:
        print(f"\n  ⚠️  WARNING: Some classes have very few samples (< {min_samples}):")
        for cls, cnt in low_count_classes:
            print(f"      '{cls}': {cnt} samples")
        print(f"    This may cause issues with stratified cross-validation.")
        print(f"    Consider collecting more data or merging small classes.")
    
    return dict(zip(unique, counts))


# ============================================================================
# ABLATION STUDY
# ============================================================================
def evaluate_subset(X_full, y, feature_names, subset_features, subset_name, 
                    n_hidden=1000, activation='tanh', C=1e-3):
    """
    Evaluate a single feature subset with cross-validation.
    """
    # Get indices for this subset
    indices = [feature_names.index(f) for f in subset_features if f in feature_names]
    
    if len(indices) == 0:
        return None
    
    X_subset = X_full[:, indices]
    
    # Scale features for ELM (important for tanh activation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    elm = ELMClassifier(n_hidden=n_hidden, activation=activation, C=C, 
                        random_state=RANDOM_STATE)
    
    scores = cross_val_score(elm, X_scaled, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    return {
        'name': subset_name,
        'n_features': len(subset_features),
        'features': subset_features,
        'cv_f1_mean': scores.mean(),
        'cv_f1_std': scores.std(),
        'cv_scores': scores.tolist()
    }


def run_ablation_study(df, labels=None, exclude_labels=None,
                       n_hidden=1000, activation='tanh', C=1e-3):
    """
    Run full ablation study across all feature subsets.
    
    Parameters
    ----------
    df : DataFrame
        Feature DataFrame from feature_extraction.py
    labels : array-like, optional
        If provided, use these labels instead of df['label']
    exclude_labels : list of str, optional
        Labels to exclude from training (e.g., ['unconclusive', 'undefined'])
    n_hidden, activation, C : ELM hyperparameters
    
    Returns
    -------
    results_df : DataFrame with results for each subset
    best_subset : dict with best subset info
    """
    
    print("=" * 70)
    print("  ELM Feature Group Ablation Study")
    print("=" * 70)
    
    # Apply label filtering
    if exclude_labels is None:
        exclude_labels = ['EXCLUDE', 'unconclusive', 'undefined', 'N/A']
    
    df_filtered, dropped = filter_labels(df, exclude_labels=exclude_labels)
    
    # Prepare data
    if labels is not None:
        # If labels provided separately, need to align with filtered df
        # For simplicity, assume labels correspond to df order
        # Better to use df's label column
        pass
    
    # Use label column from filtered dataframe
    labels = df_filtered['label'].values
    
    # Check class balance
    class_counts = check_class_balance(labels)
    
    # Drop non-feature columns
    non_feature_cols = ['sample_id', 'label', 'timestamp', 'source_file', 'is_clean_label']
    feature_cols = [c for c in df_filtered.columns if c not in non_feature_cols]
    X = df_filtered[feature_cols].values
    feature_names = feature_cols
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_
    
    print(f"\n  Dataset after filtering: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Classes: {list(class_names)}")
    print(f"  Class distribution:")
    for cls, count in zip(class_names, np.bincount(y)):
        print(f"    {cls:<20} {count:>4} samples ({100*count/len(y):.1f}%)")
    
    # Create feature subsets
    subsets, groups = create_feature_subsets(feature_names)
    
    print(f"\n  Feature groups:")
    for group_name, feats in groups.items():
        print(f"    {group_name:<20} {len(feats):>3} features")
    
    print(f"\n  Testing {len(subsets)} feature subsets...")
    print(f"  ELM config: n_hidden={n_hidden}, activation='{activation}', C={C}")
    print("\n" + "-" * 70)
    
    # Evaluate each subset
    results = []
    
    for subset_name, subset_features in subsets:
        print(f"\n  Evaluating: {subset_name} ({len(subset_features)} features)")
        start_time = time.time()
        
        result = evaluate_subset(X, y, feature_names, subset_features, subset_name,
                                 n_hidden, activation, C)
        
        if result:
            elapsed = time.time() - start_time
            print(f"    CV F1: {result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f} "
                  f"({elapsed:.2f}s)")
            results.append(result)
        else:
            print(f"    Skipped (no valid features)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cv_f1_mean', ascending=False).reset_index(drop=True)
    
    # Find best subset
    best_subset = results_df.iloc[0].to_dict()
    
    # ========================================================================
    # PLOTTING
    # ========================================================================
    plot_ablation_results(results_df, best_subset)
    
    # ========================================================================
    # DETAILED EVALUATION OF BEST SUBSET
    # ========================================================================
    print("\n" + "=" * 70)
    print("  BEST FEATURE SUBSET - DETAILED EVALUATION")
    print("=" * 70)
    
    best_features = best_subset['features']
    best_indices = [feature_names.index(f) for f in best_features]
    X_best = X[:, best_indices]
    
    # Train/test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_best, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ELM
    elm = ELMClassifier(n_hidden=n_hidden, activation=activation, C=C, 
                        random_state=RANDOM_STATE)
    elm.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = elm.predict(X_test_scaled)
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    
    print(f"\n  Best subset: {best_subset['name']}")
    print(f"  Number of features: {best_subset['n_features']}")
    print(f"\n  Test Set Performance ({len(X_test)} samples):")
    print("  " + "-" * 50)
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Confusion matrix
    plot_confusion_matrix(y_test_labels, y_pred_labels, class_names, 
                          best_subset['name'])
    
    # Save everything
    save_results(results_df, best_subset, elm, scaler, le, best_features)
    
    return results_df, best_subset


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_ablation_results(results_df, best_subset):
    """Plot comparison of all feature subsets."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: F1 scores by subset (bar chart)
    ax1 = axes[0]
    colors = ['#4C9BE8' if row['name'] != best_subset['name'] else '#E8834C' 
              for _, row in results_df.iterrows()]
    
    bars = ax1.barh(range(len(results_df)), results_df['cv_f1_mean'], 
                    xerr=results_df['cv_f1_std'], color=colors, alpha=0.8,
                    capsize=3, error_kw={'linewidth': 1})
    
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['name'])
    ax1.set_xlabel('CV F1 Score (Macro)')
    ax1.set_title('Feature Subset Performance Comparison')
    ax1.axvline(results_df['cv_f1_mean'].max(), color='gray', 
                linestyle='--', alpha=0.5)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax1.text(row['cv_f1_mean'] + 0.01, i, f"{row['cv_f1_mean']:.3f}", 
                va='center', fontsize=9)
    
    # Plot 2: F1 vs number of features
    ax2 = axes[1]
    scatter = ax2.scatter(results_df['n_features'], results_df['cv_f1_mean'],
                         c=results_df['cv_f1_mean'], cmap='viridis', 
                         s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Label points
    for _, row in results_df.iterrows():
        ax2.annotate(row['name'].replace(' ', '\n'), 
                    (row['n_features'], row['cv_f1_mean']),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('CV F1 Score (Macro)')
    ax2.set_title('Performance vs Feature Count')
    
    # Highlight best
    ax2.scatter([best_subset['n_features']], [best_subset['cv_f1_mean']],
               color='red', s=200, marker='*', edgecolors='black',
               label=f"Best: {best_subset['name']}")
    ax2.legend(loc='lower right')
    
    plt.colorbar(scatter, ax=ax2, label='F1 Score')
    
    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'feature_subset_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {PLOT_DIR / 'feature_subset_comparison.png'}")


def plot_confusion_matrix(y_true, y_pred, class_names, subset_name):
    """Plot confusion matrix for best subset."""
    
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm], 
        ["Counts", "Normalized (Recall)"], 
        ["d", ".2f"]
    ):
        disp = ConfusionMatrixDisplay(data, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=fmt)
        ax.set_title(f"{subset_name} - {title}")
        ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'best_subset_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'best_subset_confusion_matrix.png'}")


# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_results(results_df, best_subset, elm, scaler, le, best_features):
    """Save all results, model, and metadata."""
    
    # Save results DataFrame
    results_df.to_csv(PLOT_DIR / 'ablation_results.csv', index=False)
    
    # Save best subset info
    report = {
        'best_subset_name': best_subset['name'],
        'n_features': best_subset['n_features'],
        'cv_f1_mean': best_subset['cv_f1_mean'],
        'cv_f1_std': best_subset['cv_f1_std'],
        'features': best_subset['features'],
    }
    joblib.dump(report, MODEL_DIR / 'best_subset_report.pkl')
    
    # Save model and artifacts
    joblib.dump(elm, MODEL_DIR / 'elm_best_subset.pkl')
    joblib.dump(scaler, MODEL_DIR / 'scaler_best_subset.pkl')
    joblib.dump(le, MODEL_DIR / 'label_encoder.pkl')
    joblib.dump(best_features, MODEL_DIR / 'selected_features.pkl')
    
    print(f"\n  Results saved to:")
    print(f"    {PLOT_DIR}/")
    print(f"    {MODEL_DIR}/")
    
    # Print summary
    print("\n" + "=" * 70)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"\n  Best feature subset: {best_subset['name']}")
    print(f"  CV F1 Score: {best_subset['cv_f1_mean']:.4f} ± {best_subset['cv_f1_std']:.4f}")
    print(f"\n  Top 5 feature subsets:")
    print(results_df[['name', 'n_features', 'cv_f1_mean', 'cv_f1_std']].head(5).to_string(index=False))


# ============================================================================
# HYPERPARAMETER TUNING FOR BEST SUBSET (Optional)
# ============================================================================
def tune_elm_hyperparameters(X, y, feature_names, best_features, n_iter=20):
    """
    After finding the best feature subset, tune ELM hyperparameters.
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    print("\n" + "=" * 70)
    print("  HYPERPARAMETER TUNING FOR BEST FEATURE SUBSET")
    print("=" * 70)
    
    # Prepare data
    indices = [feature_names.index(f) for f in best_features]
    X_best = X[:, indices]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_best)
    
    # Parameter grid
    param_grid = {
        'n_hidden': [100, 200, 500, 1000, 2000, 3000, 5000],
        'activation': ['tanh', 'relu', 'sigmoid'],
        'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
    }
    
    elm = ELMClassifier(random_state=RANDOM_STATE)
    
    search = RandomizedSearchCV(
        elm, param_grid, n_iter=n_iter, cv=CV_FOLDS,
        scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE,
        verbose=1
    )
    
    search.fit(X_scaled, y)
    
    print(f"\n  Best parameters: {search.best_params_}")
    print(f"  Best CV F1: {search.best_score_:.4f}")
    
    # Save tuned model
    joblib.dump(search.best_estimator_, MODEL_DIR / 'elm_tuned_best_subset.pkl')
    joblib.dump(search.cv_results_, MODEL_DIR / 'tuning_results.pkl')
    
    return search


# ============================================================================
# INFERENCE HELPER
# ============================================================================
class ELMInferenceModel:
    """Wrapper for inference with the best ELM model."""
    
    def __init__(self, model, scaler, label_encoder, selected_features):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.selected_features = selected_features
    
    @classmethod
    def load(cls, folder=MODEL_DIR, tuned=False):
        folder = Path(folder)
        model_name = 'elm_tuned_best_subset.pkl' if tuned else 'elm_best_subset.pkl'
        
        model = joblib.load(folder / model_name)
        scaler = joblib.load(folder / 'scaler_best_subset.pkl')
        le = joblib.load(folder / 'label_encoder.pkl')
        features = joblib.load(folder / 'selected_features.pkl')
        
        print(f"Loaded ELM model from {folder}/")
        return cls(model, scaler, le, features)
    
    def predict(self, features_dict):
        """Predict from a feature dictionary."""
        # Extract selected features in correct order
        x = np.array([features_dict.get(f, 0) for f in self.selected_features])
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        
        y_pred = self.model.predict(x_scaled)[0]
        y_proba = self.model.predict_proba(x_scaled)[0]
        
        prediction = self.label_encoder.inverse_transform([y_pred])[0]
        probabilities = dict(zip(self.label_encoder.classes_, y_proba))
        
        return prediction, probabilities


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    from feature_extraction import process_multiple_files
    from ehda_normalization import prepare_training_data
    
    # Load data
    folder = sys.argv[1] if len(sys.argv) > 1 else "new data"
    print(f"Loading data from: {folder}")
    
    df = process_multiple_files("*.json", folder=folder)
    
    # Prepare data (without feature selection for now)
    # Note: prepare_training_data already excludes "EXCLUDE" by default
    df_raw, X, labels, feature_names, normalizer = prepare_training_data(
        df, drop_metadata=False, exclude_label="EXCLUDE"
    )
    
    # Run ablation study with ELM, also excluding "unconclusive" and "undefined"
    results_df, best_subset = run_ablation_study(
        df_raw,
        exclude_labels=['EXCLUDE', 'unconclusive', 'undefined', 'N/A'],
        n_hidden=1000,
        activation='tanh',
        C=1e-3
    )
    
    # Optional: tune hyperparameters on best subset
    print("\n" + "=" * 70)
    response = input("Run hyperparameter tuning on best subset? (y/n): ")
    if response.lower() == 'y':
        le = joblib.load(MODEL_DIR / 'label_encoder.pkl')
        y = le.transform(labels)  # Note: labels here are from filtered df_raw
        tune_elm_hyperparameters(X, y, feature_names, best_subset['features'], n_iter=30)
    
    print("\n" + "=" * 70)
    print("  DONE!")
    print("=" * 70)
    print(f"""
    To use the best model for inference:
    
        from elm_feature_ablation import ELMInferenceModel
        
        model = ELMInferenceModel.load()
        
        # Extract features from a sample
        from feature_extraction import extract_features
        features = extract_features(sample_dict)
        
        # Predict
        pred, proba = model.predict(features)
        print(f"Predicted mode: {{pred}}")
        print(f"Probabilities: {{proba}}")
    """)