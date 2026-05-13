"""
TITLE: electrospray class functions
"""

import numpy as np
import csv
import json
from scipy.signal import argrelextrema
from scipy import signal, stats
from scipy.signal import hilbert
from scipy.signal import butter, lfilter
from scipy.integrate import trapezoid

class ElectrosprayConfig:
    def __init__(self, file_setup):
        self.file_setup = file_setup
        self.ki = 6.46  # no units

    def __repr__(self):
        """dictionary = {
            "electrical conductivity": str(self.k_electrical_conductivity),
            "flow_rate": str(self.q_flow_rate),
            "voltage": str(self.voltage),
            "setup": str(self.json_setup_obj)
        }
        return "config: %s " % (json.dumps(dictionary))"""
        d = dict(
            config=dict(comment=str(self.current_comment),
                        flow_rate_min=str(self.flow_rate_min), type_of_measurement=str(self.typeofmeasurement)))
        return json.dumps(d, sort_keys=True)

    def load_json_config_setup(self):

        print("load_json_config_setup")
        # print(self.file_setup)

        with open(self.file_setup, 'r') as file:
            # First we load existing data into a dict.
            self.json_setup_obj = json.load(file)
            # print(type(self.json_setup_obj))

        """with open(self.file_setup) as json_file:
            self.json_setup_obj = json.load(json_file)"""

    def load_json_config_liquid(self, file_liquid):

        print("load_json_config_liquid")

        with open(file_liquid, 'r') as file:
            # First we load existing data into a dict.
            self.json_liquid_obj = json.load(file)
            print(self.json_liquid_obj)

        """with open(self.file_liquid) as json_file:
            self.json_liquid_obj = json.load(json_file) # dictionary"""

    def get_json_liquid(self):
        return self.json_liquid_obj

    def get_json_setup(self):
        return self.json_setup_obj

    def set_comment_current(self, current_comment):
        self.current_comment = current_comment

    def set_type_of_measurement(self, typeofmeasurement):
        self.type_of_measurement = typeofmeasurement

    def get_dict_config(self):
        dictionary = {
            "voltage regime": self.type_of_measurement
        }
        return dictionary

    def get_alpha_chen_pui(self):
        return ((self.γ * self.k_electrical_conductivity * self.q_flow_rate / self.k) ** (.5))

    # def get_cone_jet_current_est_chen_pui(self):
    # b = i_actual / ((self.γ * self.k_electrical_conductivity * self.q_flow_rate) ** .5)
    def get_flow_rate_min_ian(self):
        # print("\nconfig liquid:", self.data_dict['config']['liquid'])
        dieletric_const = self.json_liquid_obj['dielectric const']
        electrical_conductivity = self.json_liquid_obj['electrical conductivity']
        permitivity = self.json_liquid_obj['vacuum permitivity']
        surface_tension = self.json_liquid_obj['surface tension']
        rho = self.json_liquid_obj['density']
        self.flow_rate_min_ian = (permitivity * surface_tension / (rho * electrical_conductivity))
        return self.flow_rate_min_ian

    def flow_rate_min_est_chen_pui(self):
        # print("\nconfig liquid:", self.data_dict['config']['liquid'])
        dieletric_const = self.json_liquid_obj['dielectric const']
        electrical_conductivity = self.json_liquid_obj['electrical conductivity']
        permitivity = self.json_liquid_obj['vacuum permitivity']
        surface_tension = self.json_liquid_obj['surface tension']
        rho = self.json_liquid_obj['density']
        self.flow_rate_chen_pui = (
                (dieletric_const ** 0.5) * permitivity * surface_tension / (rho * electrical_conductivity))

    def get_flow_rate_min_est_chen_pui(self):
        return self.flow_rate_chen_pui

    def get_dict_flow_rate_min_est_chen_pui(self):
        dictionary = {
            "flow_rate_chen_pui": self.flow_rate_chen_pui
        }
        return dictionary

    def get_cone_jet_current_est_hartman(self, i_actual):
        b = i_actual / ((self.γ * self.k_electrical_conductivity * self.q_flow_rate) ** .5)
        # b = I_actual/pow((y * K * Q), (1/2)) 
        I_hartman = b * ((self.γ * self.k_electrical_conductivity * self.q_flow_rate) ** .5)
        b = 0.5
        I_hartman_05 = b * ((self.γ * self.k_electrical_conductivity * self.q_flow_rate) ** .5)
        b = 2
        I_hartman_2 = b * ((self.γ * self.k_electrical_conductivity * self.q_flow_rate) ** .5)
        return I_hartman, I_hartman_05, I_hartman_2


# *****************************************
#             MEASUREMENTS
# *****************************************

class ElectrosprayMeasurements:
    """ Electrospray setup representation """

    def __init__(self, name, data, voltage, flow_rate, day_measurement, current, target_voltage):
        self.name = name  # name of liquid
        self.data = data  # array nA
        self.flow_rate = flow_rate  # m3/s
        self.voltage = voltage  # Volt
        self.day_measurement = day_measurement  # date
        # self.gas_coflow_rate  = gas_coflow_rate
        self.current = current
        self.target_voltage = target_voltage

    def __repr__(self):
        dictionary = {
            "current": str(self.data.tolist()),  # array nA
            "flow_rate": self.flow_rate,
            "voltage": self.voltage,
            "current_PS": self.current,
            "date_and_time": self.day_measurement,
            "target_voltage": self.target_voltage
        }

        return (json.dumps(dictionary))

    def get_measurements_dictionary(self):
        dictionary = {
            "name": self.name,
            "current": self.data.tolist(),  # array nA
            "flow_rate": self.flow_rate,
            "voltage": str(self.voltage),
            "current_PS": str(self.current),
            "date_and_time": str(self.day_measurement),
            "target_voltage": self.target_voltage
        }
        # self.json_measurements_obj.write(json.dumps((dictionary), sort_keys=True, indent=4, separators=(". ", " = ")))
        return dictionary

    def get_flow_rate_actual(self):
        return self.flow_rate

    def get_measurements(self):
        return self.name, self.data, self.voltage,  self.flow_rate, self.impedance, self.current, self.shape_current, self.target_voltage

    def set_data(self, data_update):
        self.data = data_update  # array nA

    def set_voltage(self, voltage_update):
        self.voltage = voltage_update



# *****************************************
#             PROCESSING
# *****************************************


import numpy as np
import json
from scipy import signal
from scipy.signal import lfilter, filtfilt, argrelextrema
from scipy.integrate import trapezoid
import pywt

class ElectrosprayDataProcessing:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.clear_results()

    def clear_results(self):
        """Resets all stored values to prevent data leakage between measurements."""
        self.mean_value = 0
        self.variance = 0
        self.stddev = 0
        self.med = 0
        self.rms = 0
        self.datapoints_filtered = np.array([])
        self.psd_freqs = np.array([])
        self.psd_welch = np.array([])
        self.ml_features = {} # Stores advanced features

    # ── Filtering ─────────────────────────────────────────────────────
    def calculate_filter(self, a_coef, b_coef, datapoints):
        """Applies zero-phase filtering to avoid time-shift in features."""
        # Using filtfilt instead of lfilter prevents the 'right-shift' delay
        self.datapoints_filtered = filtfilt(b_coef, a_coef, datapoints)

    # ── Time Domain ───────────────────────────────────────────────────
    def calculate_statistics(self, data):
        self.mean_value = np.mean(data)
        self.variance = np.var(data)
        self.stddev = np.std(data)
        self.med = np.median(data)
        self.rms = np.sqrt(np.mean(data ** 2))

    def calculate_peaks_signal(self, data, threshold=39950):#Osciloscope caps out at 80 volts
        """Calculates saturation/clipping metrics (useful for ML to detect 'Out of Range')."""
        qty_max = np.sum(data >= threshold)
        pct_max = (qty_max / len(data)) * 100
        return threshold, qty_max, pct_max

    # ── Frequency Domain ──────────────────────────────────────────────
    def calculate_power_spectral_density(self, data):
        """Calculates PSD once using the high-res ML parameters (4096)."""
        # Standardizing on 4096 for both DB and ML resolution
        self.psd_freqs, self.psd_welch = signal.welch(
            data, fs=self.sample_rate, nperseg=4096, noverlap=2048, window="hann"
        )

    def calculate_band_powers(self):
        """Calculates the 5 bands required for the database."""
        bands = {
            "v_low": (0, 50), "low": (50, 500), "mid": (500, 2000),
            "high": (2000, 10000), "v_high": (10000, self.sample_rate / 2)
        }
        band_energies = {}
        for name, (low, high) in bands.items():
            idx = np.logical_and(self.psd_freqs >= low, self.psd_freqs < high)
            if np.any(idx):
                band_energies[f"band_power_{name}"] = float(trapezoid(self.psd_welch[idx], self.psd_freqs[idx]))
            else:
                band_energies[f"band_power_{name}"] = 0.0
        return band_energies
        
    # ── Advanced ML Feature Extraction ──
    def extract_advanced_ml_features(self):
        """Calculates features only needed for ML models (crest, kurtosis, entropy, wavelets)."""
        x = self.datapoints_filtered
        total_power = np.sum(self.psd_welch)
        
        # Advanced Time Domain
        peak = np.max(np.abs(x))
        self.ml_features.update({
            "peak": float(peak),
            "crest_factor": float(peak / self.rms if self.rms > 0 else 0.0),
            "kurtosis": float(stats.kurtosis(x, fisher=True)),
            "skewness": float(stats.skew(x)),
            "peak_to_peak": float(np.max(x) - np.min(x)),
            "zero_crossing_rate": float(np.sum(np.diff(np.sign(x - self.mean_value)) != 0) / len(x))
        })

        # Advanced Frequency Domain
        if total_power > 0:
            psd_norm = np.clip(self.psd_welch / total_power, 1e-12, None)
            self.ml_features.update({
                "dominant_freq": float(self.psd_freqs[np.argmax(self.psd_welch)]),
                "mean_freq": float(np.sum(self.psd_freqs * self.psd_welch) / total_power),
                "spectral_entropy": float(-np.sum(psd_norm * np.log2(psd_norm))),
                "total_power": float(total_power)
            })

        # Wavelets
        coeffs = pywt.wavedec(x, wavelet='db4', level=6)
        total_energy = sum(np.sum(c**2) for c in coeffs)
        for i, c in enumerate(coeffs):
            label = f"wt_approx_L6" if i == 0 else f"wt_detail_L{7 - i}"
            energy = np.sum(c**2)
            self.ml_features[f"{label}_energy"] = float(energy)
            self.ml_features[f"{label}_energy_rel"] = float(energy / total_energy if total_energy > 0 else 0.0)

    def get_db_features_dictionary(self):
        """Returns only the columns saved to the database."""
        bp = self.calculate_band_powers()
        return {
            "mean_na": float(self.mean_value),
            "variance_na": float(self.variance),
            "deviation_na": float(self.stddev),
            "median_na": float(self.med),
            "rms_na": float(self.rms),
            **bp
        }

    # ── Data Export ───────────────────────────────────────────────────
    def get_statistics_dictionary(self):
        # Extract band powers to include in the flat dictionary
        bp = self.calculate_band_powers()
        
        dictionary = {
            "mean": np.float64(self.mean_value),
            "variance": np.float64(self.variance),
            "deviation": np.float64(self.stddev),
            "median": np.float64(self.med),
            "rms": np.float64(self.rms),
            #"spray_mode": self.shape_current,
            #"ml_spray_mode": self.ml_shape_current,
            #"nn_spray_mode": self.nn_shape_current,
            **bp  # Merges the band power dictionary into this one
        }
        return dictionary

    def __repr__(self):
        d = self.get_statistics_dictionary()
        # Convert values to strings for JSON serialization
        return json.dumps({k: str(v) for k, v in d.items()}, sort_keys=True)