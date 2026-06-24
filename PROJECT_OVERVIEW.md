# Electrospray AI Classification Project Overview

## Project Description
This project focuses on training an Artificial Intelligence model to classify different modes of electrospray (e.g., `cone_jet`, `dripping`, `intermittent`, `multi_jet`). It utilizes both Random Forest (RF) and XGBoost algorithms to process and classify sample data.

### How It Works
1. **Data Ingestion**: The system loads data via an `ElectrosprayDatabase`. It cross-references metadata in the database (like operating conditions: target voltage, actual voltage, flow rate, etc.) with raw `.npy` waveform files containing sensor readings.
2. **Feature Extraction**: Raw waveform data is filtered (e.g., using a low-pass butterworth filter with a 3,000 Hz cutoff) and processed into a vector of 66 mathematical features. These features describe the signal's shape, amplitude, frequency domain (via Power Spectral Density), and wavelet transformations.
3. **Multi-Strategy Normalization**: Because absolute electrical current can vary drastically based on solution conductivity, a two-layer normalization system (`ehda_normalization.py`) is applied:
   - **Signal-Level (Layer 1)**: Z-score or robust scaling on the raw waveform.
   - **Feature-Level (Layer 2)**: Custom normalization with 5 different strategies applied to the feature matrix (Linear, Log+Robust, Robust, Passthrough, Low-Variance). This ensures all inputs are balanced and mathematically comparable before hitting the AI model.
4. **Training**: Features and manual classification labels are fed into the training script (`train_code.py`), where the normalizer is fitted, and the model weights are generated and saved for live inference.

---

## Recent Modifications & Enhancements

### 1. Feature Visualization (`plot_features.py`)
To better understand how specific features contribute to the AI's decision-making, a new Python script was created. This script:
- Replicates the data extraction and normalization pipeline without triggering a full training session.
- Iterates over all generated features and plots them using `seaborn.stripplot`.
- Creates side-by-side scatter plots mapping the **Raw** vs. **Normalized** values, colored/clustered by their classification label.
- Output plots are saved as PNG files into `current_classification/plots/`.

### 2. Safeguarding the Training Pipeline (`train_code.py`)
Previously, `train_code.py` executed its training logic directly at the top level. This caused issues when trying to import utility functions from the script elsewhere, as it would accidentally trigger the entire training loop and fail due to misaligned working directories.
- **Fix**: The main execution block in `train_code.py` was safely wrapped inside an `if __name__ == '__main__':` condition. This adheres to Python best practices and allows other files to safely import dependencies from it.

### 3. Granular Data Export (`export_dominant_freq.py`)
A specialized export script was developed to extract specific feature data for external analysis (e.g., in Excel or external charting software).
- It runs the feature extraction pipeline and targets the `dominant_freq` feature.
- It pairs the original raw value, the newly normalized value, the target label, and the sample ID.
- The results are packaged and exported securely to a `dominant_freq_values.csv` file.
