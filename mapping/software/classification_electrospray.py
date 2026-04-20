"""
Electrospray spray mode classification.

Three classifiers are available:
  1. do_classification()        – Sjaak rule-based (unchanged)
  2. do_ml_classification()     – New hierarchical classifier (66 features)
  3. do_nn_classification()     – Kept for compatibility, routes to hierarchical

The new ML classifier uses the full feature extraction + normalization pipeline.
Models are loaded once at startup and reused for every measurement point.

Feature vector: 66 signal features extracted from the raw 50k-sample waveform.
  (see feature_extraction.py for full list)
"""

import numpy as np


SPRAY_MODES = {0: "Cone Jet", 1: "Corona", 2: "Dripping",
               3: "Intermittent", 4: "Multi Jet"}


class ElectrosprayClassification:

    def __init__(self):
        self.previous_states = []
        self.cone_jet_mean   = 0
        self._SAFE_EPSILON   = 1e-323

    # ──────────────────────────────────────────────────────────────────
    #  Rule-based classifier  (Sjaak — unchanged)
    # ──────────────────────────────────────────────────────────────────

    def do_classification(self, mean, median, stddeviation,
                          psd_values, variance,
                          max_value_of_the_data, quantity_max_data,
                          percentage_max, flow_rate,
                          fft_max_peaks_array, cont_fft_max_peaks,
                          I_chen_pui=None):

        classification = "Undefined"

        try:
            std_over_mean    = (stddeviation / mean) if mean   != 0 else 0
            mean_over_median = (mean / median)       if median != 0 else 0

            if mean <= 5:
                if std_over_mean > 2.5:
                    if mean_over_median < 0.9 or mean_over_median > 1.1:
                        classification = "Dripping"

            if mean > 5:
                if 0.5 < std_over_mean < 2.5:
                    if mean_over_median < 0.9 or mean_over_median > 1.1:
                        classification = "Intermittent"

            if mean > 5:
                if std_over_mean < 0.5:
                    if 0.9 < mean_over_median < 1.1:
                        classification = "Cone Jet"

        except Exception as e:
            print(f"[CLASSIFIER] Sjaak error: {e}")

        try:
            if percentage_max >= 0.5:
                classification = "Corona"
        except Exception as e:
            print(f"[CLASSIFIER] Corona check error: {e}")

        self.previous_states.append(classification)
        return classification

    # ──────────────────────────────────────────────────────────────────
    #  New ML classifier  (hierarchical, 66-feature pipeline)
    # ──────────────────────────────────────────────────────────────────

    def do_ml_classification(self, model, mean, variance, std_dev,
                             median, rms, voltage, flow_rate,
                             datapoints=None) -> str:
        """
        Parameters
        ----------
        model      : EHDAModels instance (from load_ml_models())
                     Falls back to legacy 9-feature model if datapoints=None
        datapoints : raw current array (50,000 samples) — required for new pipeline
        voltage    : actual voltage [V]
        flow_rate  : flow rate [µL/min]

        Returns
        -------
        Human-readable mode string, e.g. "cone_jet" or "stable_transitioning"
        """
        # ── New pipeline (full waveform available) ─────────────────────
        if datapoints is not None and hasattr(model, "is_new_pipeline"):
            try:
                return model.classify(
                    datapoints  = datapoints,
                    voltage     = voltage,
                    flow_rate   = flow_rate,
                    mean        = mean,
                    std         = std_dev,
                    median      = median,
                    rms         = rms,
                    variance    = variance,
                )
            except Exception as e:
                print(f"[CLASSIFIER] New ML error: {e}")
                return "N/A"

        # ── Legacy fallback (9-feature vector, old models) ────────────
        try:
            safe_median   = median if median != 0 else self._SAFE_EPSILON
            features      = [mean, variance, std_dev, median, rms,
                             float(voltage), float(flow_rate),
                             mean / safe_median, std_dev / safe_median]
            code = model.predict([features])
            return SPRAY_MODES.get(int(code), "Undefined")
        except Exception as e:
            print(f"[CLASSIFIER] Legacy ML error: {e}")
            return "N/A"

    def do_nn_classification(self, model, mean, variance, std_dev,
                             median, rms, voltage, flow_rate,
                             datapoints=None) -> str:
        """
        Routes to the new pipeline if model is EHDAModels,
        otherwise falls back to the original NN behaviour.
        """
        if datapoints is not None and hasattr(model, "is_new_pipeline"):
            # Both ML and NN slots now use the same hierarchical classifier.
            # nn_classification will carry the confidence / flag string.
            try:
                return model.classify_detail(
                    datapoints = datapoints,
                    voltage    = voltage,
                    flow_rate  = flow_rate,
                    mean       = mean,
                    std        = std_dev,
                    median     = median,
                    rms        = rms,
                    variance   = variance,
                )
            except Exception as e:
                print(f"[CLASSIFIER] New NN error: {e}")
                return "N/A"

        # Legacy NN fallback
        try:
            safe_median = median if median != 0 else self._SAFE_EPSILON
            features    = [mean, variance, std_dev, median, rms,
                           float(voltage), float(flow_rate),
                           mean / safe_median, std_dev / safe_median]
            result = model.predict([features])
            if isinstance(result[0], str):
                return result[0]
            return SPRAY_MODES.get(int(result[0]), "Undefined")
        except Exception as e:
            print(f"[CLASSIFIER] Legacy NN error: {e}")
            return "N/A"