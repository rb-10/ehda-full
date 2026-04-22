"""
DIAGNOSTIC SCRIPT: Test Feature Extraction & Normalization
===========================================================

This script tests your JSON data structure and helps identify issues
with feature extraction and normalization before running the full pipeline.

USAGE:
    python diagnose_reclassify.py --input_file your_json_file.json --model_folder path/to/models
"""

import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Tuple
import traceback

try:
    from feature_extraction import extract_features
    from ehda_normalization import EHDAFeatureNormalizer
    from ehda_classifier import EHDAClassifier
except ImportError as e:
    print(f"ERROR: Could not import required modules. {e}")
    sys.exit(1)


def parse_json_sample(json_obj: dict) -> Tuple[dict, np.ndarray]:
    """Parse a single sample from a JSON object."""
    metadata = {}
    
    standard_fields = [
        'id', 'timestamp', 'target_voltage', 'flow_rate', 'rf_spray_mode',
        'xgb_spray_mode', 'current_PS', 'mean', 'deviation', 'median',
        'rms', 'variance', 'image_classification'
    ]
    
    for field in standard_fields:
        if field in json_obj:
            metadata[field] = json_obj[field]
    
    if 'current' not in json_obj:
        raise ValueError("Sample does not contain 'current' field")
    
    current_data = json_obj['current']
    
    if isinstance(current_data, list):
        current_array = np.array(current_data, dtype=np.float64)
    else:
        raise ValueError(f"Expected 'current' to be a list, got {type(current_data)}")
    
    return metadata, current_array


def diagnose(input_file: str, model_folder: str):
    """Run diagnostic checks on your data and model."""
    
    print("="*70)
    print("DIAGNOSTIC SCRIPT: EHDA Reclassification")
    print("="*70)
    
    # 1. Check JSON file
    print("\n[1] CHECKING JSON FILE")
    print("-" * 70)
    
    json_path = Path(input_file)
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        return False
    
    print(f"File: {json_path.name}")
    print(f"File size: {json_path.stat().st_size} bytes")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print("[OK] JSON file loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load JSON: {e}")
        return False
    
    # 2. Find samples
    print("\n[2] SCANNING FOR SAMPLES")
    print("-" * 70)
    
    sample_keys = [k for k in json_data.keys() if k.startswith('sample')]
    print(f"Found {len(sample_keys)} samples")
    
    if not sample_keys:
        print("ERROR: No samples found in JSON file")
        return False
    
    # 3. Test feature extraction on first sample
    print("\n[3] TESTING FEATURE EXTRACTION")
    print("-" * 70)
    
    sample_key = sample_keys[0]
    sample_obj = json_data[sample_key]
    
    print(f"Testing on sample: {sample_key}")
    
    try:
        metadata, current_array = parse_json_sample(sample_obj)
        print(f"[OK] Sample parsed")
        print(f"  Current array shape: {current_array.shape}")
        print(f"  Current array dtype: {current_array.dtype}")
        print(f"  Current array range: [{current_array.min():.4f}, {current_array.max():.4f}]")
    except Exception as e:
        print(f"ERROR: Failed to parse sample: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Wrap array in dictionary since extract_features expects dict format
        sample_dict = {"current": current_array}
        features_dict = extract_features(sample_dict)
        print(f"[OK] Features extracted")
        print(f"  Number of features: {len(features_dict)}")
        print(f"  Feature names ({len(features_dict)} total):")
        for i, (name, value) in enumerate(features_dict.items(), 1):
            print(f"    {i:2d}. {name:<40} = {value}")
        print(f"\n  Feature values sample: {list(features_dict.values())[:5]}")
    except Exception as e:
        print(f"ERROR: Feature extraction failed: {e}")
        traceback.print_exc()
        return False
    
    # 4. Test normalization
    print("\n[4] TESTING NORMALIZATION")
    print("-" * 70)
    
    model_path = Path(model_folder)
    if not model_path.exists():
        print(f"ERROR: Model folder not found: {model_path}")
        return False
    
    try:
        normalizer_path = model_path / ".." / "scalers"
        normalizer = EHDAFeatureNormalizer.load(str(normalizer_path))
        print(f"[OK] Normalizer loaded from {normalizer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load normalizer: {e}")
        traceback.print_exc()
        return False
    
    # 4b. Load selected features if available
    print("\n[4b] CHECKING FOR SELECTED FEATURES")
    print("-" * 70)
    
    selected_feature_names = None
    try:
        from feature_selector import load_selected_features
        selected_feature_names = load_selected_features(str(model_path.parent))
        print(f"[OK] Loaded {len(selected_feature_names)} selected features")
        print(f"  Original features extracted: {len(features_dict)}")
        print(f"  Will use only selected features for prediction")
    except FileNotFoundError:
        print(f"[INFO] No selected_features.pkl found")
        print(f"  Will use all {len(features_dict)} extracted features")
    except Exception as e:
        print(f"[INFO] Could not load selected features: {e}")
        print(f"  Will use all {len(features_dict)} extracted features")
    
    # Subset to selected features if available
    if selected_feature_names:
        features_dict = {n: features_dict[n] for n in selected_feature_names if n in features_dict}
        print(f"  Features after subsetting: {len(features_dict)}")
    
    try:
        features_df = pd.DataFrame([features_dict])
        print(f"\n[OK] Features DataFrame created")
        print(f"  DataFrame shape: {features_df.shape}")
        print(f"  DataFrame columns: {list(features_df.columns)[:5]}... ({len(features_df.columns)} total)")
    except Exception as e:
        print(f"ERROR: Failed to create DataFrame: {e}")
        traceback.print_exc()
        return False
    
    try:
        features_normalized_df = normalizer.transform(features_df)
        print(f"[OK] Features normalized")
        print(f"  Normalized DataFrame shape: {features_normalized_df.shape}")
        print(f"  Normalized DataFrame type: {type(features_normalized_df)}")
        print(f"  Normalized DataFrame dtypes:\n{features_normalized_df.dtypes}")
    except Exception as e:
        print(f"ERROR: Normalization failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        x_normalized = features_normalized_df.values.flatten()
        print(f"[OK] Features converted to numpy array")
        print(f"  Array shape: {x_normalized.shape}")
        print(f"  Array dtype: {x_normalized.dtype}")
        print(f"  Array values sample: {x_normalized[:5]}")
    except Exception as e:
        print(f"ERROR: Failed to convert to array: {e}")
        traceback.print_exc()
        return False
    
    # 5. Test classifier
    print("\n[5] TESTING CLASSIFIER")
    print("-" * 70)
    
    try:
        classifier = EHDAClassifier.load(str(model_path), model_name='random_forest')
        print(f"[OK] Classifier loaded")
    except Exception as e:
        print(f"ERROR: Failed to load classifier: {e}")
        traceback.print_exc()
        return False
    
    try:
        prediction, probabilities = classifier.predict(x_normalized)
        print(f"[OK] Prediction successful")
        print(f"  Predicted class: {prediction}")
        print(f"  Confidence: {max(probabilities.values()):.4f}")
        print(f"  All probabilities: {probabilities}")
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        traceback.print_exc()
        return False
    
    # 6. Test on multiple samples
    print("\n[6] TESTING ON MULTIPLE SAMPLES")
    print("-" * 70)
    
    success_count = 0
    error_count = 0
    
    for i, sample_key in enumerate(sample_keys[:10]):  # Test first 10 samples
        try:
            sample_obj = json_data[sample_key]
            metadata, current_array = parse_json_sample(sample_obj)
            # Wrap array in dictionary since extract_features expects dict format
            sample_dict = {"current": current_array}
            features_dict = extract_features(sample_dict)
            features_df = pd.DataFrame([features_dict])
            features_normalized_df = normalizer.transform(features_df)
            x_normalized = features_normalized_df.values.flatten()
            prediction, probabilities = classifier.predict(x_normalized)
            
            success_count += 1
            status = "OK"
        except Exception as e:
            error_count += 1
            status = f"FAILED: {str(e)[:50]}"
        
        print(f"  {sample_key:>10} ... [{status}]")
    
    print(f"\nResults: {success_count} succeeded, {error_count} failed out of {min(10, len(sample_keys))} tested")
    
    # Final summary
    print("\n" + "="*70)
    if error_count == 0:
        print("DIAGNOSIS: All tests passed! Your setup should work.")
    else:
        print(f"DIAGNOSIS: {error_count} test(s) failed. Check errors above.")
    print("="*70)
    
    return error_count == 0


if __name__ == "__main__":
    import argparse
    
    DEFAULT_INPUT_FILE = "data/DMF/Current/experiment_0.json"
    DEFAULT_MODEL_FOLDER = "current_classification/models"
    
    parser = argparse.ArgumentParser(
        description="Diagnose EHDA reclassification issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Use default files
  python diagnose_reclassify.py
  
  # Specify custom files
  python diagnose_reclassify.py --input_file path/to/file.json --model_folder path/to/models
        """
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to a JSON file to test (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        '--model_folder',
        type=str,
        default=DEFAULT_MODEL_FOLDER,
        help=f"Path to model folder (default: {DEFAULT_MODEL_FOLDER})"
    )
    
    args = parser.parse_args()
    
    success = diagnose(args.input_file, args.model_folder)
    sys.exit(0 if success else 1)