"""
EHDA JSON Reclassification Pipeline
====================================
Loads a trained EHDA model and reclassifies all JSON files in a folder.

USAGE:
    python reclassify_json_files.py --input_folder /path/to/json/files --model_folder models/ --output_folder results/

Or without arguments:
    python reclassify_json_files.py
    (defaults: input_folder = 'new data', model_folder = 'models', output_folder = 'results')

WHAT IT DOES:
  1. Scans for all *.json files in input_folder
  2. Extracts features from each sample (same pipeline as training)
  3. Normalizes features using the saved normalizer
  4. Predicts spray mode using the trained model
  5. Saves updated JSON files with new predictions
  6. Generates a summary CSV with all predictions and confidence scores

OUTPUT:
  - results/reclassified_json/ — updated JSON files with new predictions
  - results/reclassification_summary.csv — table of all predictions
  - results/reclassification_log.txt — detailed processing log

REQUIREMENTS:
    pip install scikit-learn pandas numpy scipy pywt joblib tqdm
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
import sys
from pathlib import Path
from typing import Tuple, Dict, List
from tqdm import tqdm
import logging

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS FROM YOUR MODULES
# ─────────────────────────────────────────────────────────────────────────────
# These assume feature_extraction, ehda_normalization, and ehda_classifier
# are in the same directory or in your Python path
try:
    from feature_extraction import extract_features
    from ehda_normalization import EHDAFeatureNormalizer
    from ehda_classifier import EHDAClassifier
except ImportError as e:
    print(f"ERROR: Could not import required modules. {e}")
    print("Make sure feature_extraction.py, ehda_normalization.py, and ehda_classifier.py are in the same directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_FOLDER = "data/DMF/Current"
DEFAULT_MODEL_FOLDER = "current_classification/models"
DEFAULT_OUTPUT_FOLDER = "data/DMF/Current/results"


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging to file and console with UTF-8 support."""
    logger = logging.getLogger("ehda_reclassify")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler (UTF-8 encoding to support all characters)
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler (UTF-8 encoding for Windows console compatibility)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    if hasattr(ch, 'encoding'):
        ch.encoding = 'utf-8'
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_json_sample(json_obj: dict) -> Tuple[dict, np.ndarray]:
    """
    Parse a single sample from a JSON object.
    
    Returns:
        (metadata_dict, current_array)
    """
    metadata = {}
    
    # Standard fields that appear in sample objects
    standard_fields = [
        'id', 'timestamp', 'target_voltage', 'flow_rate', 'rf_spray_mode',
        'xgb_spray_mode', 'current_PS', 'mean', 'deviation', 'median',
        'rms', 'variance', 'image_classification'
    ]
    
    for field in standard_fields:
        if field in json_obj:
            metadata[field] = json_obj[field]
    
    # Extract current signal (should be a list of numbers)
    if 'current' not in json_obj:
        raise ValueError("Sample does not contain 'current' field")
    
    current_data = json_obj['current']
    
    # Handle case where current is stored as a list
    if isinstance(current_data, list):
        current_array = np.array(current_data, dtype=np.float64)
    else:
        raise ValueError(f"Expected 'current' to be a list, got {type(current_data)}")
    
    return metadata, current_array


def load_json_files(folder: Path) -> List[Tuple[Path, dict]]:
    """
    Scan folder for all *.json files and load them.
    
    Returns:
        List of (file_path, json_data) tuples
    """
    json_files = list(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {folder}")
    
    data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            data.append((json_file, json_data))
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {json_file}: {e}")
            continue
    
    return data


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION & PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def reclassify_sample(
    json_obj: dict,
    classifier: "EHDAClassifier",
    normalizer: "EHDAFeatureNormalizer",
    logger: logging.Logger,
) -> Dict:
    """
    Extract features from a JSON sample and predict its spray mode.
    
    Returns:
        Dictionary with original data + new predictions
    """
    result = {
        'success': False,
        'original_label': None,
        'predicted_label': None,
        'confidence': None,
        'all_probabilities': {},
        'error': None,
        'metadata': {}
    }
    
    try:
        # Parse the JSON sample
        metadata, current_array = parse_json_sample(json_obj)
        result['metadata'] = metadata
        result['original_label'] = metadata.get('rf_spray_mode', 'N/A')
        
        # Extract features - wrap array in dictionary since extract_features expects dict format
        sample_dict = {"current": current_array}
        features_dict = extract_features(sample_dict)
        
        # Subset to selected features if available
        if hasattr(reclassify_sample, '_selected_features'):
            selected_names = reclassify_sample._selected_features
            features_dict = {n: features_dict[n] for n in selected_names if n in features_dict}
        
        # Convert to DataFrame for normalization
        features_df = pd.DataFrame([features_dict])
        
        # Normalize using the trained normalizer
        # The normalizer.transform() returns a DataFrame, so get the values properly
        features_normalized_df = normalizer.transform(features_df)
        
        # Convert the normalized DataFrame to a numpy array (1D for prediction)
        x_normalized = features_normalized_df.values.flatten()
        
        # Predict using the classifier
        prediction, probabilities = classifier.predict(x_normalized)
        
        result['success'] = True
        result['predicted_label'] = prediction
        result['all_probabilities'] = probabilities
        
        # Find max probability (confidence)
        result['confidence'] = max(probabilities.values())
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        logger.warning(f"Error processing sample: {e}")
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FILE WRITING
# ─────────────────────────────────────────────────────────────────────────────
def save_reclassified_json(
    original_json: dict,
    prediction_result: dict,
    output_path: Path
) -> None:
    """
    Save the original JSON with added prediction fields.
    
    Adds the following fields to each sample:
      - 'predicted_spray_mode': the model's prediction
      - 'predicted_confidence': confidence score (0-1)
      - 'predicted_probabilities': dict of all class probabilities
      - 'original_rf_spray_mode': backup of the original label
    """
    updated_json = json.loads(json.dumps(original_json))  # Deep copy
    
    # Find sample keys in the JSON (usually "sample 0", "sample 1", etc.)
    sample_keys = [k for k in updated_json.keys() if k.startswith('sample')]
    
    # For now, update the first sample (adjust logic if multiple samples per file)
    if sample_keys:
        sample_key = sample_keys[0]
        
        if prediction_result['success']:
            updated_json[sample_key]['predicted_spray_mode'] = prediction_result['predicted_label']
            updated_json[sample_key]['predicted_confidence'] = float(prediction_result['confidence'])
            updated_json[sample_key]['predicted_probabilities'] = prediction_result['all_probabilities']
        else:
            updated_json[sample_key]['predicted_spray_mode'] = 'ERROR'
            updated_json[sample_key]['predicted_error'] = prediction_result['error']
    
    # Write to output file (UTF-8 encoding)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_json, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main(
    input_folder: str = DEFAULT_INPUT_FOLDER,
    model_folder: str = DEFAULT_MODEL_FOLDER,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    model_name: str = "random_forest"
):
    """
    Main reclassification pipeline.
    
    Parameters
    ----------
    input_folder : str
        Path to folder containing JSON files to reclassify
    model_folder : str
        Path to folder containing trained model artifacts
    output_folder : str
        Path where results will be saved
    model_name : str
        Which model to use: "random_forest", "xgboost", or "elm"
    """
    
    # Setup paths and logging
    input_path = Path(input_folder)
    model_path = Path(model_folder)
    output_path = Path(output_folder)
    json_output_path = output_path / "reclassified_json"
    
    output_path.mkdir(parents=True, exist_ok=True)
    json_output_path.mkdir(parents=True, exist_ok=True)
    
    log_path = output_path / "reclassification_log.txt"
    logger = setup_logging(log_path)
    
    logger.info("="*70)
    logger.info("EHDA JSON RECLASSIFICATION PIPELINE")
    logger.info("="*70)
    logger.info(f"Input folder:      {input_path.resolve()}")
    logger.info(f"Model folder:      {model_path.resolve()}")
    logger.info(f"Output folder:     {output_path.resolve()}")
    logger.info(f"Model to use:      {model_name}")
    
    # Validate paths
    if not input_path.exists():
        logger.error(f"Input folder not found: {input_path}")
        return False
    
    if not model_path.exists():
        logger.error(f"Model folder not found: {model_path}")
        return False
    
    # Load trained model and normalizer
    try:
        logger.info("\nLoading trained model...")
        classifier = EHDAClassifier.load(str(model_path), model_name=model_name)
        logger.info(f"[OK] Loaded {model_name} classifier")
        
        logger.info("Loading normalizer...")
        normalizer_path = model_path / ".." / "scalers"
        normalizer = EHDAFeatureNormalizer.load(str(normalizer_path))
        logger.info("[OK] Loaded normalizer")
        
    except FileNotFoundError as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        return False
    
    # Load selected features if available
    selected_features = None
    try:
        from feature_selector import load_selected_features
        selected_features = load_selected_features(str(model_path.parent))
        logger.info(f"[OK] Loaded {len(selected_features)} selected features")
        # Store in function for use in reclassify_sample
        reclassify_sample._selected_features = selected_features
    except FileNotFoundError:
        logger.info("[INFO] No selected_features.pkl found — using all extracted features")
    except Exception as e:
        logger.warning(f"[WARNING] Could not load selected features: {e}")
    
    # Load JSON files
    try:
        logger.info(f"\nScanning for JSON files in {input_path}...")
        json_files = load_json_files(input_path)
        logger.info(f"Found {len(json_files)} JSON files")
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    
    # Process each file
    results_list = []
    
    logger.info("\n" + "="*70)
    logger.info("RECLASSIFYING SAMPLES")
    logger.info("="*70)
    
    for file_path, json_data in tqdm(json_files, desc="Processing files"):
        logger.info(f"\nProcessing: {file_path.name}")
        
        # Find sample keys
        sample_keys = [k for k in json_data.keys() if k.startswith('sample')]
        
        if not sample_keys:
            logger.warning(f"  No samples found in {file_path.name}")
            continue
        
        # Reclassify each sample in the file
        for sample_key in sample_keys:
            sample_obj = json_data[sample_key]
            
            # Make prediction
            result = reclassify_sample(sample_obj, classifier, normalizer, logger)
            
            # Store result with file info
            result['file'] = file_path.name
            result['sample_key'] = sample_key
            result['sample_id'] = sample_obj.get('id', 'N/A')
            results_list.append(result)
            
            # Log the result (use ASCII-safe indicators)
            if result['success']:
                logger.info(f"  [OK] {sample_key}: {result['original_label']} -> {result['predicted_label']} "
                           f"(confidence: {result['confidence']:.2%})")
            else:
                logger.warning(f"  [FAILED] {sample_key}: {result['error']}")
        
        # Save reclassified JSON
        output_json_path = json_output_path / file_path.name
        save_reclassified_json(json_data, results_list[-1], output_json_path)
        logger.info(f"  Saved: {output_json_path}")
    
    # Create summary DataFrame and CSV
    logger.info("\n" + "="*70)
    logger.info("GENERATING SUMMARY")
    logger.info("="*70)
    
    summary_data = []
    for r in results_list:
        row = {
            'file': r['file'],
            'sample_key': r['sample_key'],
            'sample_id': r['sample_id'],
            'original_label': r['original_label'],
            'predicted_label': r['predicted_label'] if r['success'] else 'ERROR',
            'confidence': r['confidence'] if r['success'] else None,
            'status': 'success' if r['success'] else 'failed',
            'error': r['error'] if not r['success'] else None,
        }
        
        # Add individual class probabilities
        if r['success'] and r['all_probabilities']:
            for class_name, prob in r['all_probabilities'].items():
                row[f'prob_{class_name}'] = prob
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_path / "reclassification_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    
    logger.info(f"\n[OK] Saved summary to {summary_csv_path}")
    
    # Print statistics
    success_count = (summary_df['status'] == 'success').sum()
    error_count = (summary_df['status'] == 'failed').sum()
    
    logger.info(f"\n  Total samples processed: {len(summary_df)}")
    logger.info(f"  Successful predictions: {success_count}")
    logger.info(f"  Failed predictions:     {error_count}")
    
    if success_count > 0:
        avg_confidence = summary_df[summary_df['status'] == 'success']['confidence'].mean()
        logger.info(f"  Average confidence:     {avg_confidence:.2%}")
        
        # Show distribution of predicted classes
        logger.info("\n  Predicted spray mode distribution:")
        pred_dist = summary_df[summary_df['status'] == 'success']['predicted_label'].value_counts()
        for label, count in pred_dist.items():
            logger.info(f"    {label:<20} {count:>5} ({count/success_count:>6.1%})")
    
    logger.info("\n" + "="*70)
    logger.info("RECLASSIFICATION COMPLETE")
    logger.info("="*70)
    
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reclassify EHDA JSON files using a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default folders
  python reclassify_json_files.py
  
  # Specify custom folders
  python reclassify_json_files.py --input_folder data/ --output_folder predictions/
  
  # Use different model (default is random_forest)
  python reclassify_json_files.py --model_name xgboost
        """
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        default=DEFAULT_INPUT_FOLDER,
        help=f"Folder containing JSON files (default: {DEFAULT_INPUT_FOLDER})"
    )
    
    parser.add_argument(
        '--model_folder',
        type=str,
        default=DEFAULT_MODEL_FOLDER,
        help=f"Folder containing trained model (default: {DEFAULT_MODEL_FOLDER})"
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Where to save results (default: {DEFAULT_OUTPUT_FOLDER})"
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='random_forest',
        choices=['random_forest', 'xgboost', 'elm'],
        help="Which trained model to use (default: random_forest)"
    )
    
    args = parser.parse_args()
    
    success = main(
        input_folder=args.input_folder,
        model_folder=args.model_folder,
        output_folder=args.output_folder,
        model_name=args.model_name
    )
    
    sys.exit(0 if success else 1)