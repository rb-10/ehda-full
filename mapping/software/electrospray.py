"""
TITLE: electrospray class functions
"""

import numpy as np
import csv
import json
from scipy.signal import argrelextrema
from scipy import signal
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
        self.fourier_transform = np.array([])
        self.freq = np.array([])
        self.psd_freqs = np.array([])
        self.psd_welch = np.array([])
        
        self.fourier_peaks = []
        self.all_fourier_peaks = []
        
        # Classification labels
        self.shape_current = ""
        self.generalist_ml_shape_current = ""
        self.ml_shape_current = ""
        self.nn_shape_current = ""

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

    def calculate_peaks_signal(self, data, threshold=3995.0):
        """Calculates saturation/clipping metrics (useful for ML to detect 'Out of Range')."""
        qty_max = np.sum(data >= threshold)
        pct_max = (qty_max / len(data)) * 100
        return threshold, qty_max, pct_max

    # ── Frequency Domain ──────────────────────────────────────────────
    def calculate_fft_raw(self, datapoints):
        """Standard FFT for high-resolution frequency analysis."""
        n = datapoints.size
        self.fourier_transform = np.fft.rfft(datapoints)
        self.freq = np.fft.rfftfreq(n, d=1/self.sample_rate)

    def calculate_power_spectral_density(self, data):
        """
        Welch's Method: Smoothes the spectrum by averaging windowed segments.
        Better for ML as it reduces variance in frequency features.
        """
        # nperseg=1024 is standard, but you can adjust based on RECORD_LENGTH
        self.psd_freqs, self.psd_welch = signal.welch(data, fs=self.sample_rate, nperseg=1024)
        return self.psd_freqs, self.psd_welch

    def calculate_band_powers(self):
        if self.psd_welch.size == 0:
            return {}

        bands = {
            "v_low":  (0, 50),
            "low":    (50, 500),
            "mid":    (500, 2000),
            "high":   (2000, 10000),
            "v_high": (10000, self.sample_rate / 2)
        }

        band_energies = {}
        for name, (low, high) in bands.items():
            idx = np.logical_and(self.psd_freqs >= low, self.psd_freqs < high)

            # Check if we have enough points to integrate
            if np.any(idx):
                band_energies[f"band_power_{name}"] = trapezoid(self.psd_welch[idx], self.psd_freqs[idx])
            else:
                band_energies[f"band_power_{name}"] = 0.0

        return band_energies
        
        
    def calculate_fft_peaks(self, min_freq=50, min_amp=1500):
        """Consolidated peak finder to identify dominant oscillation frequencies."""
        mag = np.abs(self.fourier_transform)
        # Identify local maxima
        indices = argrelextrema(mag, np.greater, order=5)[0]
        
        # Filter for physical relevance
        mask = (self.freq[indices] > min_freq) & (mag[indices] > min_amp)
        valid_idx = indices[mask]
        
        # Sort by amplitude descending
        valid_idx = valid_idx[np.argsort(mag[valid_idx])[::-1]]
        
        self.fourier_peaks = []
        for i, idx in enumerate(valid_idx[:3]):
            rank = ["1st", "2nd", "3rd"][i]
            self.fourier_peaks.append(f"{rank}: ")
            self.fourier_peaks.append([float(mag[idx]), float(self.freq[idx])])
            
        return self.fourier_peaks, len(valid_idx)

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
            "spray_mode": self.shape_current,
            "ml_spray_mode": self.ml_shape_current,
            "nn_spray_mode": self.nn_shape_current,
            **bp  # Merges the band power dictionary into this one
        }
        return dictionary

    def __repr__(self):
        d = self.get_statistics_dictionary()
        # Convert values to strings for JSON serialization
        return json.dumps({k: str(v) for k, v in d.items()}, sort_keys=True)