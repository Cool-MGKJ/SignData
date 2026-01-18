"""
Train All Models: SVM, Random Forest, and Gradient Boosting

This script trains all three machine learning models on the same dataset
and saves them for comparison and visualization.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Model paths
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SVM_MODEL_PATH = MODELS_DIR / "asl_svm_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "asl_rf_model.pkl"
GB_MODEL_PATH = MODELS_DIR / "asl_gb_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.pkl"

# Dataset path
DEFAULT_DATASET_PATH = "data/hand_sign_singlehand.json"

# Training configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# SVM hyperparameters
SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_KERNEL = "rbf"

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None

# Gradient Boosting hyperparameters
GB_N_ESTIMATORS = 200
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH = 3
GB_SUBSAMPLE = 0.9


def load_datasets(dataset_paths: List[str]) -> Tuple[List[dict], List[str], int, int, int]:
    """
    Load samples from multiple JSON dataset files.
    
    Returns:
        Tuple of (features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len)
    """
    all_features = []
    all_labels = []
    max_hit_order_len = 0
    max_chain_code_len = 0
    max_palm_angles_len = 0

    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset file not found: {dataset_path}")
            continue

        print(f"Loading dataset: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Handle array format
        all_samples = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and "samples" in entry:
                    all_samples.extend(entry.get("samples", []))
                elif isinstance(entry, dict) and "label" in entry:
                    all_samples.append(entry)
        else:
            if isinstance(data, dict) and "samples" in data:
                all_samples = data.get("samples", [])

        for sample in all_samples:
            label = sample.get("label")
            if not label:
                continue

            # Extract points (single hand: left OR right, whichever is present)
            points_flat = []
            points_left = sample.get("points_left", [])
            points_right = sample.get("points_right", [])

            # Use first non-empty hand
            points_source = points_right if points_right else points_left
            
            if points_source:
                for point_dict in points_source:
                    if isinstance(point_dict, dict):
                        points_flat.extend([point_dict["x"], point_dict["y"], point_dict["z"]])
            
            if not points_flat:
                continue

            # Extract hit_order and chain_code
            hit_order = sample.get("hit_order", [])
            chain_code = sample.get("chain_code", [])
            
            # Extract palm_angles from both hands and flatten
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
            
            # Track maximum lengths
            max_hit_order_len = max(max_hit_order_len, len(hit_order))
            max_chain_code_len = max(max_chain_code_len, len(chain_code))
            max_palm_angles_len = max(max_palm_angles_len, len(palm_angles_flat))

            # Store features
            all_features.append({
                "points_flat": points_flat,
                "hit_order": hit_order,
                "chain_code": chain_code,
                "palm_angles_flat": palm_angles_flat
            })
            all_labels.append(label)

    return all_features, all_labels, max_hit_order_len, max_chain_code_len, max_palm_angles_len


def prepare_features(
    features_list: List[dict], 
    labels_list: List[str],
    max_hit_order_len: int,
    max_chain_code_len: int,
    max_palm_angles_len: int
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Prepare features for training.
    
    Returns:
        Tuple of (X_scaled, y_encoded, scaler, label_encoder)
    """
    POINTS_FEATURES = 63  # Single hand
    
    padded_features = []
    
    for feat_dict in features_list:
        points_flat = feat_dict["points_flat"]
        hit_order = feat_dict.get("hit_order", [])
        chain_code = feat_dict.get("chain_code", [])
        palm_angles_flat = feat_dict.get("palm_angles_flat", [])
        
        # Pad/truncate points_flat to 63 features
        if len(points_flat) < POINTS_FEATURES:
            points_padded = list(points_flat) + [0.0] * (POINTS_FEATURES - len(points_flat))
        elif len(points_flat) > POINTS_FEATURES:
            points_padded = list(points_flat[:POINTS_FEATURES])
        else:
            points_padded = list(points_flat)
        
        # Pad hit_order
        hit_order_padded = list(hit_order[:max_hit_order_len]) + [0] * (max_hit_order_len - len(hit_order))
        
        # Pad chain_code
        chain_code_padded = list(chain_code[:max_chain_code_len]) + [0] * (max_chain_code_len - len(chain_code))
        
        # Pad palm_angles_flat
        palm_angles_padded = list(palm_angles_flat[:max_palm_angles_len]) + [0.0] * (max_palm_angles_len - len(palm_angles_flat))
        
        # Combine
        combined = points_padded + hit_order_padded + chain_code_padded + palm_angles_padded
        padded_features.append(combined)
    
    X = np.array(padded_features, dtype=np.float32)
    y = np.array(labels_list)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nFeature matrix shape: {X_scaled.shape}")
    print(f"  - Points (landmarks, single hand): 63 (21 x 3)")
    print(f"  - Hit order: {max_hit_order_len}")
    print(f"  - Chain code: {max_chain_code_len}")
    print(f"  - Palm angles: {max_palm_angles_len}")
    print(f"  - Total features: {63 + max_hit_order_len + max_chain_code_len + max_palm_angles_len}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return X_scaled, y_encoded, scaler, label_encoder


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """Train SVM model."""
    print("\n" + "=" * 60)
    print("Training SVM (RBF kernel)...")
    print("=" * 60)
    
    svm_model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=True,
        random_state=RANDOM_STATE,
    )
    
    svm_model.fit(X_train, y_train)
    print("SVM training completed.")
    
    return svm_model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest model."""
    print("\n" + "=" * 60)
    print("Training Random Forest...")
    print("=" * 60)
    
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")
    
    # Print feature importance summary
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    print("\nTop 10 most important features (Random Forest):")
    for idx in top_indices:
        print(f"  Feature {idx:3d}: {importances[idx]:.6f}")
    
    return rf_model


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """Train Gradient Boosting model."""
    print("\n" + "=" * 60)
    print("Training Gradient Boosting...")
    print("=" * 60)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        learning_rate=GB_LEARNING_RATE,
        max_depth=GB_MAX_DEPTH,
        subsample=GB_SUBSAMPLE,
        random_state=RANDOM_STATE,
    )
    
    gb_model.fit(X_train, y_train)
    print("Gradient Boosting training completed.")
    
    return gb_model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str, label_encoder: LabelEncoder) -> Dict:
    """Evaluate a model and return metrics."""
    print("\n" + "=" * 60)
    print(f"Evaluating {model_name}...")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    # Confusion matrix
    unique_classes = sorted(set(np.concatenate([y_test, y_pred])))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    if len(unique_classes) > 0:
        class_names = [label_encoder.classes_[idx] for idx in unique_classes]
        report = classification_report(
            y_test, y_pred, labels=unique_classes, target_names=class_names, zero_division=0
        )
        print("\nClassification Report:")
        print(report)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def save_artifacts(
    svm_model: SVC,
    rf_model: RandomForestClassifier,
    gb_model: GradientBoostingClassifier,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    max_hit_order_len: int,
    max_chain_code_len: int,
    max_palm_angles_len: int,
) -> None:
    """Save all models and preprocessing artifacts."""
    print("\n" + "=" * 60)
    print("Saving models and artifacts...")
    print("=" * 60)
    
    # Save models
    joblib.dump(svm_model, SVM_MODEL_PATH)
    print(f"Saved SVM model to: {SVM_MODEL_PATH}")
    
    joblib.dump(rf_model, RF_MODEL_PATH)
    print(f"Saved Random Forest model to: {RF_MODEL_PATH}")
    
    joblib.dump(gb_model, GB_MODEL_PATH)
    print(f"Saved Gradient Boosting model to: {GB_MODEL_PATH}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to: {SCALER_PATH}")
    
    # Save label encoder
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")
    
    # Save feature configuration
    feature_config = {
        "max_hit_order_len": max_hit_order_len,
        "max_chain_code_len": max_chain_code_len,
        "max_palm_angles_len": max_palm_angles_len,
    }
    joblib.dump(feature_config, FEATURE_CONFIG_PATH)
    print(f"Saved feature config to: {FEATURE_CONFIG_PATH}")
    
    print("\nAll artifacts saved successfully!")


def print_class_distribution(labels_list: List[str]) -> None:
    """Print class distribution."""
    from collections import Counter
    
    label_counts = Counter(labels_list)
    print("\n" + "=" * 60)
    print("Class Distribution:")
    print("=" * 60)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples")
    print(f"\nTotal: {len(labels_list)} samples")


def main(dataset_paths: List[str] = None) -> None:
    """Main training pipeline."""
    if dataset_paths is None:
        dataset_paths = [DEFAULT_DATASET_PATH]
    
    print("=" * 60)
    print("ASL Sign Classification Training Pipeline - All Models")
    print("=" * 60)
    
    # Step 1: Load datasets
    print("\nStep 1: Loading datasets...")
    features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len = load_datasets(dataset_paths)
    
    # Print class distribution
    print_class_distribution(labels_list)
    
    # Load feature config if available (to match training)
    if FEATURE_CONFIG_PATH.exists():
        feature_config = joblib.load(FEATURE_CONFIG_PATH)
        max_hit_order_len = feature_config.get("max_hit_order_len", max_hit_order_len)
        max_chain_code_len = feature_config.get("max_chain_code_len", max_chain_code_len)
        max_palm_angles_len = feature_config.get("max_palm_angles_len", max_palm_angles_len)
        print(f"\nUsing saved feature config: hit_order={max_hit_order_len}, chain_code={max_chain_code_len}, palm_angles={max_palm_angles_len}")
    
    # Step 2: Prepare features
    print("\nStep 2: Preparing features...")
    X_scaled, y_encoded, scaler, label_encoder = prepare_features(
        features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len
    )
    
    # Step 3: Split data
    print("\nStep 3: Splitting data into training and test sets...")
    
    from collections import Counter
    class_counts = Counter(y_encoded)
    min_samples_per_class = min(class_counts.values())
    
    if min_samples_per_class < 2:
        print(f"Warning: Some classes have only 1 sample. Using non-stratified split.")
        use_stratify = False
    else:
        use_stratify = True
    
    if len(X_scaled) < 10:
        print(f"Warning: Very small dataset ({len(X_scaled)} samples). Using all data for training.")
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y_encoded, y_encoded
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
            stratify=y_encoded if use_stratify else None
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Train all models
    print("\nStep 4: Training models...")
    svm_model = train_svm(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Step 5: Evaluate all models
    print("\nStep 5: Evaluating models...")
    svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM", label_encoder)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", label_encoder)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting", label_encoder)
    
    # Step 6: Save artifacts
    save_artifacts(svm_model, rf_model, gb_model, scaler, label_encoder, 
                   max_hit_order_len, max_chain_code_len, max_palm_angles_len)
    
    # Step 7: Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"\nSVM:")
    print(f"  Accuracy: {svm_metrics['accuracy']:.4f} ({svm_metrics['accuracy']*100:.2f}%)")
    print(f"\nRandom Forest:")
    print(f"  Accuracy: {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.2f}%)")
    print(f"\nGradient Boosting:")
    print(f"  Accuracy: {gb_metrics['accuracy']:.4f} ({gb_metrics['accuracy']*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)
    print(f"\nSaved files:")
    print(f"  - {SVM_MODEL_PATH}")
    print(f"  - {RF_MODEL_PATH}")
    print(f"  - {GB_MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {LABEL_ENCODER_PATH}")
    print(f"  - {FEATURE_CONFIG_PATH}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_paths = sys.argv[1:]
        main(dataset_paths)
    else:
        main()
