"""
current_classification/classify.py
───────────────────────────────────
Shared classification function used by:
  - main_electrospray.py
  - main_electrospray_5_measurements.py
  - current_classification/reclassify.py

Pipeline (in order):
  0. Compute ratio_to_previous_step from DB
  1. Extract advanced ML features
  2. Build unified feature vector
  3. Normalize
  4. Align to each model's expected feature set
  5. ML predictions (RF + XGB)
  6. Hard-coded Rule 1 — multi_jet propagation  (lower priority)
  7. Hard-coded Rule 2 — corona override         (higher priority, applied last)

Hard-coded rules
────────────────
Rule 1 — multi_jet propagation
  If the most-recent previous voltage step (different target_voltage, same
  flow_rate) was classified as multi_jet (rf_spray_mode starts with
  "multi_jet"), override both RF and XGB results to "multi_jet (rule)".
  This rule is SKIPPED if no earlier step exists at a different voltage for
  the same flow_rate (i.e., the current step is the first step of the sweep).

Rule 2 — corona override
  If pct_max > 0.5, override both RF and XGB results to "corona (rule)".
  Rule 2 takes priority over Rule 1.

ID resolution
─────────────
  Live pipelines : result dict has no "id" yet → uses db.get_last_id()
  reclassify.py  : result dict IS the DB record and carries "id"
                   → uses result["id"] for correct per-record lookups
"""

from __future__ import annotations


# ── Thresholds ────────────────────────────────────────────────────────────────

PCT_MAX_CORONA_THRESHOLD = 0.5   # Rule 2: pct_max above this → corona


# ── Main classification function ──────────────────────────────────────────────

def classify_sample(processing, result: dict, ml_models: dict, db) -> tuple[str, str]:
    """
    Classify a single measurement and apply hard-coded override rules.

    Parameters
    ----------
    processing : ElectrosprayDataProcessing
        Processing object with filter/statistics already computed.
    result : dict
        Measurement result dict.  In live pipelines this is the unsaved
        result; in reclassify.py it is the full DB record (includes "id").
    ml_models : dict
        Dict with keys "rf", "xgb", "normalizer".
    db : ElectrosprayDatabase
        Database instance used for previous-step look-ups.

    Returns
    -------
    (rf_result, xgb_result) : tuple[str, str]
        Final classification strings, e.g. "cone_jet (87%)" or
        "corona (rule)" or "multi_jet (rule)".
    """
    if not ml_models:
        return "N/A", "N/A"

    try:
        from current_classification.ehda_normalization import prepare_inference_sample
        import pandas as pd

        # ── 0. Resolve ID ─────────────────────────────────────────────────────
        # Live pipelines: result has no "id" yet — use last saved row.
        # reclassify.py: result IS the DB record and carries "id".
        current_id = result.get("id") if result.get("id") is not None else db.get_last_id()

        # ── 0b. ratio_to_previous_step ────────────────────────────────────────
        ratio_to_previous_step = 1.0
        if current_id is not None:
            prev_mean = db.get_previous_step_mean_na(
                current_id=current_id,
                target_voltage=float(result["target_voltage"]),
                flow_rate=float(result["flow_rate"]),
            )
            if prev_mean not in (None, 0):
                ratio_to_previous_step = float(processing.mean_value) / prev_mean

        # ── 1. Extract advanced ML features ───────────────────────────────────
        processing.extract_advanced_ml_features(ratio_to_previous_step=ratio_to_previous_step)

        # ── 2. Build unified feature vector ───────────────────────────────────
        # NOTE: solution_name / hv_position / sample_rate / n_samples are
        # included here so the live pipeline and reclassify see identical
        # feature vectors (fixes Issue #2 from the coherence audit).
        all_features = processing.get_db_features_dictionary()
        all_features.update(processing.ml_features)
        all_features.update({
            "actual_voltage":  float(result["actual_voltage"]),
            "target_voltage":  float(result["target_voltage"]),
            "flow_rate":       float(result["flow_rate"]),
            "voltage_error":   float(result["actual_voltage"]) - float(result["target_voltage"]),
            "deviation_ratio": float(
                processing.stddev / processing.mean_value
                if processing.mean_value != 0 else 0.0
            ),
            "solution_name": result.get("solution_name"),
            "hv_position":   result.get("hv_position"),
            "sample_rate":   float(result.get("sample_rate") or processing.sample_rate),
            "n_samples":     int(result.get("n_samples") or len(processing.datapoints_filtered)),
        })

        # ── 3. Normalization pipeline ──────────────────────────────────────────
        x_norm = prepare_inference_sample(all_features, ml_models["normalizer"])

        # ── 4. Align to each model's feature set ──────────────────────────────
        all_feature_names = ml_models["normalizer"].get_feature_columns()
        df_full = pd.DataFrame([x_norm], columns=all_feature_names)

        # ── 5. ML predictions ─────────────────────────────────────────────────
        rf_result = "N/A"
        if "rf" in ml_models:
            rf_features  = ml_models["rf"].feature_names
            x_rf         = df_full[rf_features].values[0]
            pred, proba  = ml_models["rf"].predict(x_rf)
            rf_result    = f"{pred} ({proba.get(pred, 0.0):.0%})"

        xgb_result = "N/A"
        if "xgb" in ml_models:
            xgb_features = ml_models["xgb"].feature_names
            x_xgb        = df_full[xgb_features].values[0]
            pred, proba  = ml_models["xgb"].predict(x_xgb)
            xgb_result   = f"{pred} ({proba.get(pred, 0.0):.0%})"

        # ── 6. Rule 1 — multi_jet propagation (lower priority) ────────────────
        # If the previous voltage step (different target_voltage, same
        # flow_rate) was classified as multi_jet, carry that label forward.
        # get_previous_step_classification returns None when no earlier step
        # exists at a different voltage (first step of the sweep), so the
        # rule is naturally skipped for the first step.
        if current_id is not None:
            prev_class = db.get_previous_step_classification(
                current_id=current_id,
                target_voltage=float(result["target_voltage"]),
                flow_rate=float(result["flow_rate"]),
            )
            if prev_class is not None and str(prev_class).startswith("multi_jet"):
                rf_result  = "multi_jet (rule)"
                xgb_result = "multi_jet (rule)"
                

        # ── 7. Rule 2 — corona override (higher priority, applied last) ───────
        # pct_max is in the result dict for both live and reclassify contexts.
        pct_max = float(result.get("pct_max") or 0.0)
        if pct_max > PCT_MAX_CORONA_THRESHOLD:
            rf_result  = "corona (rule)"
            xgb_result = "corona (rule)"

        return rf_result, xgb_result

    except Exception as e:
        print(f"[CLASSIFY] Error: {e}")
        return "error", "error"
