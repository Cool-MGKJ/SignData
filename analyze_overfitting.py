"""
Analyze Model Overfitting

This script evaluates models on both training and test sets to detect overfitting.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Model paths
MODELS_DIR = Path("models")
SVM_MODEL_PATH = MODELS_DIR / "asl_svm_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "asl_rf_model.pkl"
GB_MODEL_PATH = MODELS_DIR / "asl_gb_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.pkl"

DEFAULT_DATASET_PATH = "data/hand_sign_singlehand.json"


def load_datasets(dataset_paths: List[str]) -> Tuple[List[dict], List[str], int, int, int]:
    """Load samples from dataset files."""
    all_features = []
    all_labels = []
    max_hit_order_len = 0
    max_chain_code_len = 0
    max_palm_angles_len = 0

    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            continue
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        all_samples = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and "samples" in entry:
                    all_samples.extend(entry.get("samples", []))
                elif isinstance(entry, dict) and "label" in entry:
                    all_samples.append(entry)

        for sample in all_samples:
            label = sample.get("label")
            if not label:
                continue

            points_flat = []
            points_left = sample.get("points_left", [])
            points_right = sample.get("points_right", [])
            points_source = points_right if points_right else points_left
            
            if points_source:
                for point_dict in points_source:
                    if isinstance(point_dict, dict):
                        points_flat.extend([point_dict["x"], point_dict["y"], point_dict["z"]])
            
            if not points_flat:
                continue

            hit_order = sample.get("hit_order", [])
            chain_code = sample.get("chain_code", [])
            palm_angles_left = sample.get("palm_angles_left", [])
            palm_angles_right = sample.get("palm_angles_right", [])
            
            palm_angles_flat = []
            if palm_angles_left:
                for angle_array in palm_angles_left:
                    if isinstance(angle_array, list) and len(angle_array) == 3:
                        palm_angles_flat.extend(angle_array)
            if palm_angles_right:
                for angle_array in palm_angles_right:
                    if isinstance(angle_array, list) and len(angle_array) == 3:
                        palm_angles_flat.extend(angle_array)
            
            max_hit_order_len = max(max_hit_order_len, len(hit_order))
            max_chain_code_len = max(max_chain_code_len, len(chain_code))
            max_palm_angles_len = max(max_palm_angles_len, len(palm_angles_flat))

            all_features.append({
                "points_flat": points_flat,
                "hit_order": hit_order,
                "chain_code": chain_code,
                "palm_angles_flat": palm_angles_flat
            })
            all_labels.append(label)

    return all_features, all_labels, max_hit_order_len, max_chain_code_len, max_palm_angles_len


def prepare_features(features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len):
    """Prepare features."""
    POINTS_FEATURES = 63
    padded_features = []
    
    for feat_dict in features_list:
        points_flat = feat_dict["points_flat"]
        hit_order = feat_dict.get("hit_order", [])
        chain_code = feat_dict.get("chain_code", [])
        palm_angles_flat = feat_dict.get("palm_angles_flat", [])
        
        if len(points_flat) < POINTS_FEATURES:
            points_padded = list(points_flat) + [0.0] * (POINTS_FEATURES - len(points_flat))
        elif len(points_flat) > POINTS_FEATURES:
            points_padded = list(points_flat[:POINTS_FEATURES])
        else:
            points_padded = list(points_flat)
        
        hit_order_padded = list(hit_order[:max_hit_order_len]) + [0] * (max_hit_order_len - len(hit_order))
        chain_code_padded = list(chain_code[:max_chain_code_len]) + [0] * (max_chain_code_len - len(chain_code))
        palm_angles_padded = list(palm_angles_flat[:max_palm_angles_len]) + [0.0] * (max_palm_angles_len - len(palm_angles_flat))
        
        combined = points_padded + hit_order_padded + chain_code_padded + palm_angles_padded
        padded_features.append(combined)
    
    X = np.array(padded_features, dtype=np.float32)
    y = np.array(labels_list)
    
    return X, y


def analyze_overfitting():
    """Analyze model overfitting by comparing train vs test performance."""
    
    print("=" * 60)
    print("Overfitting Analysis")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading dataset...")
    features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len = load_datasets([DEFAULT_DATASET_PATH])
    
    # Load feature config
    if FEATURE_CONFIG_PATH.exists():
        feature_config = joblib.load(FEATURE_CONFIG_PATH)
        max_hit_order_len = feature_config.get("max_hit_order_len", max_hit_order_len)
        max_chain_code_len = feature_config.get("max_chain_code_len", max_chain_code_len)
        max_palm_angles_len = feature_config.get("max_palm_angles_len", max_palm_angles_len)
    
    # Load scaler and encoder first (to match training)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    # Prepare features
    print("2. Preparing features...")
    X, y = prepare_features(features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len)
    
    # Scale features
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    
    # Split data
    print("3. Splitting data...")
    from collections import Counter
    class_counts = Counter(y_encoded)
    use_stratify = min(class_counts.values()) >= 2
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42,
        stratify=y_encoded if use_stratify else None
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(X_scaled)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_scaled.shape[1]}")
    print(f"  Classes: {len(label_encoder.classes_)}")
    print(f"  Samples per class (train): {dict(Counter(label_encoder.inverse_transform(y_train)))}")
    print(f"  Samples per class (test): {dict(Counter(label_encoder.inverse_transform(y_test)))}")
    
    # Calculate complexity metrics
    print(f"\nComplexity Analysis:")
    samples_per_feature = len(X_train) / X_train.shape[1]
    print(f"  Training samples per feature: {samples_per_feature:.2f}")
    print(f"  (Rule of thumb: Need at least 5-10 samples per feature)")
    
    if samples_per_feature < 5:
        print(f"  WARNING: Too few samples per feature! High overfitting risk.")
    elif samples_per_feature < 10:
        print(f"  CAUTION: Borderline samples per feature. Moderate overfitting risk.")
    else:
        print(f"  OK: Sufficient samples per feature.")
    
    # Load and evaluate models
    print(f"\n4. Evaluating models on both training and test sets...")
    models = {
        "SVM": SVM_MODEL_PATH,
        "Random Forest": RF_MODEL_PATH,
        "Gradient Boosting": GB_MODEL_PATH
    }
    
    results = []
    
    for model_name, model_path in models.items():
        if not model_path.exists():
            continue
        
        model = joblib.load(model_path)
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Calculate overfitting gap
        overfitting_gap = train_accuracy - test_accuracy
        
        results.append({
            "model": model_name,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "overfitting_gap": overfitting_gap
        })
        
        print(f"\n{model_name}:")
        print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Overfitting Gap:   {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
        
        if overfitting_gap > 0.10:
            print(f"  STATUS: Significant overfitting (>10% gap)")
        elif overfitting_gap > 0.05:
            print(f"  STATUS: Moderate overfitting (5-10% gap)")
        elif overfitting_gap > 0.02:
            print(f"  STATUS: Minor overfitting (2-5% gap)")
        else:
            print(f"  STATUS: No significant overfitting")
    
    # Summary analysis
    print("\n" + "=" * 60)
    print("Overfitting Analysis Summary")
    print("=" * 60)
    
    print(f"\nPotential Causes of 100% Accuracy:")
    print(f"  1. Dataset Size:")
    print(f"     - Only {len(X_scaled)} total samples for {X_scaled.shape[1]} features")
    print(f"     - {len(X_test)} test samples may be too small to detect generalization issues")
    print(f"     - Rule of thumb: Need 1000+ samples for reliable model evaluation")
    
    print(f"\n  2. Feature-to-Sample Ratio:")
    print(f"     - {X_scaled.shape[1]} features vs {len(X_train)} training samples")
    print(f"     - Ratio: {samples_per_feature:.2f} samples per feature")
    print(f"     - Recommended: At least 10 samples per feature")
    
    print(f"\n  3. Class Separation:")
    print(f"     - If classes are very distinct, perfect separation is possible")
    print(f"     - But this may not generalize to new data with variations")
    
    print(f"\n  4. Model Complexity:")
    print(f"     - Random Forest: 100 trees (very flexible)")
    print(f"     - Gradient Boosting: 200 sequential trees (extremely flexible)")
    print(f"     - SVM RBF: Can memorize training data with high-dimensional features")
    
    print(f"\nRecommendations:")
    print(f"  1. Collect more diverse data (different people, lighting, angles)")
    print(f"  2. Use cross-validation instead of single train/test split")
    print(f"  3. Reduce model complexity:")
    print(f"     - Random Forest: Reduce n_estimators, add max_depth limit")
    print(f"     - Gradient Boosting: Increase learning_rate, reduce n_estimators")
    print(f"     - SVM: Increase C regularization or reduce gamma")
    print(f"  4. Feature selection: Remove less important features")
    print(f"  5. Data augmentation: Create variations of existing samples")
    print(f"  6. Add noise/dropout during training to improve generalization")


if __name__ == "__main__":
    analyze_overfitting()
