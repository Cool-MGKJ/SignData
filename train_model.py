"""
Machine Learning Training Pipeline for ASL Sign Classification.

This module trains SVM and Random Forest classifiers on normalized 3D hand
landmark data collected by the dataset collection tool.

Training Pipeline:
1. Load multiple JSON dataset files
2. Extract features (points_flat) and labels
3. Encode class labels
4. Apply feature scaling (StandardScaler)
5. Split data into training and test sets (stratified)
6. Train SVM and Random Forest models
7. Evaluate both models
8. Save models, scaler, and label encoder to disk
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
import joblib


# Constants
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SVM_MODEL_PATH = MODELS_DIR / "asl_svm_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "asl_rf_model.pkl"
GB_MODEL_PATH = MODELS_DIR / "asl_gb_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.pkl"

DEFAULT_DATASET_PATHS = [
    "data/asl_dataset.json",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# SVM hyperparameters
SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_KERNEL = "rbf"

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_RANDOM_STATE = RANDOM_STATE


def load_datasets(dataset_paths: List[str]) -> Tuple[List[dict], List[str], int, int, int]:
    """
    Load samples from multiple JSON dataset files.

    Args:
        dataset_paths: List of paths to JSON dataset files

    Returns:
        Tuple of (features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len) where:
        - features_list: List of dicts with 'points_flat', 'hit_order', 'chain_code', 'palm_angles_flat', 'trigger_distance'
        - labels_list: List of string labels
        - max_hit_order_len: Maximum length of hit_order across all samples
        - max_chain_code_len: Maximum length of chain_code across all samples
        - max_palm_angles_len: Maximum length of flattened palm_angles across all samples

    Raises:
        FileNotFoundError: If a dataset file doesn't exist
        ValueError: If a dataset file is malformed or missing required fields
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

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both array format and object format
            if isinstance(data, list):
                # If data is a list, it might be [{"metadata": {...}, "samples": [...]}, ...]
                samples = []
                for item in data:
                    if isinstance(item, dict) and "samples" in item:
                        samples.extend(item.get("samples", []))
                    elif isinstance(item, dict) and "label" in item:
                        # Direct sample format
                        samples.append(item)
            else:
                # Object format with "samples" key
                samples = data.get("samples", [])
            
            if not samples:
                print(f"Warning: No samples found in {dataset_path}")
                continue

            for sample in samples:
                # Extract label
                label = sample.get("label")
                if not label:
                    print(f"Warning: Sample missing label, skipping: {sample.get('id', 'unknown')}")
                    continue

                # Extract points from numbered format (points_left/points_right or points)
                points_flat = []
                
                # Try new format first: points_left and points_right
                points_left = sample.get("points_left", [])
                points_right = sample.get("points_right", [])
                
                if points_left or points_right:
                    # Convert numbered format to flat format for both hands
                    if points_left and isinstance(points_left[0], dict):
                        for point_dict in points_left:
                            points_flat.extend([point_dict["x"], point_dict["y"], point_dict["z"]])
                    
                    if points_right and isinstance(points_right[0], dict):
                        for point_dict in points_right:
                            points_flat.extend([point_dict["x"], point_dict["y"], point_dict["z"]])
                else:
                    # Try old format: points_flat or points
                    points_flat = sample.get("points_flat")
                    if not points_flat:
                        points = sample.get("points", [])
                        if points and isinstance(points[0], dict):
                            # Convert numbered format to flat format
                            points_flat = []
                            for point_dict in points:
                                points_flat.extend([point_dict["x"], point_dict["y"], point_dict["z"]])
                
                if not points_flat:
                    print(f"Warning: Sample missing points, skipping: ID={sample.get('id', 'unknown')}")
                    continue

                # Extract hit_order and chain_code
                hit_order = sample.get("hit_order", [])
                chain_code = sample.get("chain_code", [])
                
                # Extract palm_angles (pitch, yaw, roll) from both hands and flatten
                palm_angles_left = sample.get("palm_angles_left", [])
                palm_angles_right = sample.get("palm_angles_right", [])
                
                # Flatten all palm angles: [pitch, yaw, roll] for each entry
                palm_angles_flat = []
                if palm_angles_left:
                    for angle_array in palm_angles_left:
                        if isinstance(angle_array, list) and len(angle_array) == 3:
                            palm_angles_flat.extend(angle_array)  # [pitch, yaw, roll]
                
                if palm_angles_right:
                    for angle_array in palm_angles_right:
                        if isinstance(angle_array, list) and len(angle_array) == 3:
                            palm_angles_flat.extend(angle_array)  # [pitch, yaw, roll]
                
                # Extract trigger_distance from both hands and combine
                trigger_distance_left = sample.get("trigger_distance_left", [])
                trigger_distance_right = sample.get("trigger_distance_right", [])
                trigger_distance = []
                if trigger_distance_left:
                    trigger_distance.extend(trigger_distance_left)
                if trigger_distance_right:
                    trigger_distance.extend(trigger_distance_right)
                
                # Track maximum lengths for padding
                max_hit_order_len = max(max_hit_order_len, len(hit_order))
                max_chain_code_len = max(max_chain_code_len, len(chain_code))
                max_palm_angles_len = max(max_palm_angles_len, len(palm_angles_flat))

                # Store features as dict for later padding
                all_features.append({
                    "points_flat": points_flat,
                    "hit_order": hit_order,
                    "chain_code": chain_code,
                    "palm_angles_flat": palm_angles_flat,
                    "trigger_distance": trigger_distance
                })
                all_labels.append(label)

            print(f"Loaded {len(samples)} samples from {dataset_path}")

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {dataset_path}: {e}")
            continue
        except KeyError as e:
            print(f"Error: Missing required field in {dataset_path}: {e}")
            continue
        except Exception as e:
            print(f"Error loading {dataset_path}: {e}")
            continue

    if not all_features:
        raise ValueError("No valid samples loaded from any dataset file")

    print(f"\nTotal samples loaded: {len(all_features)}")
    print(f"Max hit_order length: {max_hit_order_len}")
    print(f"Max chain_code length: {max_chain_code_len}")
    print(f"Max palm_angles length: {max_palm_angles_len}")
    return all_features, all_labels, max_hit_order_len, max_chain_code_len, max_palm_angles_len


def prepare_features(
    features_list: List[dict], 
    labels_list: List[str],
    max_hit_order_len: int,
    max_chain_code_len: int,
    max_palm_angles_len: int
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Prepare features and labels for training.

    Args:
        features_list: List of dicts with 'points_flat', 'hit_order', 'chain_code', 'palm_angles_flat', 'trigger_distance'
        labels_list: List of string labels
        max_hit_order_len: Maximum length for hit_order padding
        max_chain_code_len: Maximum length for chain_code padding
        max_palm_angles_len: Maximum length for palm_angles_flat padding

    Returns:
        Tuple of (X_scaled, y_encoded, scaler, label_encoder) where:
        - X_scaled: Scaled feature matrix (n_samples, n_features)
        - y_encoded: Encoded label array (n_samples,)
        - scaler: Fitted StandardScaler
        - label_encoder: Fitted LabelEncoder
    """
    # Pad features to consistent length
    padded_features = []
    max_trigger_distance_len = 0
    max_points_len = 0
    
    # First pass: find max lengths
    for feat_dict in features_list:
        points_flat = feat_dict["points_flat"]
        trigger_distance = feat_dict.get("trigger_distance", [])
        max_points_len = max(max_points_len, len(points_flat))
        max_trigger_distance_len = max(max_trigger_distance_len, len(trigger_distance))
    
    # Standardize to 126 features for points (both hands: 21 landmarks × 3 coords × 2 hands)
    POINTS_FEATURES = 126  # Both hands
    
    for feat_dict in features_list:
        points_flat = feat_dict["points_flat"]
        hit_order = feat_dict.get("hit_order", [])
        chain_code = feat_dict.get("chain_code", [])
        palm_angles_flat = feat_dict.get("palm_angles_flat", [])
        trigger_distance = feat_dict.get("trigger_distance", [])
        
        # Pad/truncate points_flat to fixed size (126 for both hands)
        if len(points_flat) < POINTS_FEATURES:
            # Pad with zeros if only one hand
            points_padded = list(points_flat) + [0.0] * (POINTS_FEATURES - len(points_flat))
        elif len(points_flat) > POINTS_FEATURES:
            # Truncate if somehow more than both hands
            points_padded = list(points_flat[:POINTS_FEATURES])
        else:
            points_padded = list(points_flat)
        
        # Pad hit_order
        hit_order_padded = list(hit_order[:max_hit_order_len]) + [0] * (max_hit_order_len - len(hit_order))
        
        # Pad chain_code
        chain_code_padded = list(chain_code[:max_chain_code_len]) + [0] * (max_chain_code_len - len(chain_code))
        
        # Pad palm_angles_flat
        palm_angles_padded = list(palm_angles_flat[:max_palm_angles_len]) + [0.0] * (max_palm_angles_len - len(palm_angles_flat))
        
        # Pad trigger_distance
        trigger_distance_padded = list(trigger_distance[:max_trigger_distance_len]) + [0.0] * (max_trigger_distance_len - len(trigger_distance))
        
        # Combine: points_flat + hit_order + chain_code + palm_angles_flat + trigger_distance
        combined = points_padded + hit_order_padded + chain_code_padded + palm_angles_padded + trigger_distance_padded
        padded_features.append(combined)
    
    # Convert to numpy arrays
    X = np.array(padded_features, dtype=np.float32)
    y = np.array(labels_list)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nFeature matrix shape: {X_scaled.shape}")
    print(f"  - Points (landmarks, both hands): 126 (21 × 3 × 2)")
    print(f"  - Hit order: {max_hit_order_len}")
    print(f"  - Chain code: {max_chain_code_len}")
    print(f"  - Palm angles (flattened): {max_palm_angles_len}")
    print(f"  - Trigger distance: {max_trigger_distance_len}")
    print(f"  - Total features: {X_scaled.shape[1]}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")

    return X_scaled, y_encoded, scaler, label_encoder


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Train a Support Vector Machine classifier with RBF kernel.

    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training labels (n_samples,)

    Returns:
        Trained SVC model
    """
    print("\n" + "=" * 60)
    print("Training SVM (RBF kernel)...")
    print("=" * 60)

    svm_model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=True,  # Enable probability estimates
        random_state=RANDOM_STATE,
    )

    svm_model.fit(X_train, y_train)
    print("SVM training completed.")

    return svm_model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training labels (n_samples,)

    Returns:
        Trained RandomForestClassifier model
    """
    print("\n" + "=" * 60)
    print("Training Random Forest...")
    print("=" * 60)

    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,  # Use all available cores
    )

    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")

    # Print feature importance summary
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    print("\nTop 10 most important features (Random Forest):")
    for idx in top_indices:
        landmark_idx = idx // 3
        coord = ["x", "y", "z"][idx % 3]
        print(f"  Feature {idx:2d} (Landmark {landmark_idx:2d} {coord}): {importances[idx]:.6f}")

    return rf_model


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained GradientBoostingClassifier model
    """
    print("\n" + "=" * 60)
    print("Training Gradient Boosting...")
    print("=" * 60)

    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=RANDOM_STATE,
    )
    gb_model.fit(X_train, y_train)

    print("Gradient Boosting training completed.")
    return gb_model


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str, label_encoder: LabelEncoder
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained classifier model
        X_test: Test feature matrix (n_samples, n_features)
        y_test: Test labels (n_samples,)
        model_name: Name of the model (for logging)
        label_encoder: LabelEncoder used to encode labels

    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 60)
    print(f"Evaluating {model_name}...")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Print metrics
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Confusion matrix
    unique_classes = sorted(set(np.concatenate([y_test, y_pred])))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    print("\nConfusion Matrix:")
    if len(unique_classes) <= 10:
        # Print with class names for small number of classes
        class_names_short = [label_encoder.classes_[idx] for idx in unique_classes]
        print("Classes:", class_names_short)
    print(cm)

    # Classification report
    print("\nClassification Report:")
    # Only include classes that appear in test set or predictions
    unique_classes = sorted(set(np.concatenate([y_test, y_pred])))
    if len(unique_classes) > 0:
        class_names = [label_encoder.classes_[idx] for idx in unique_classes]
        report = classification_report(
            y_test, y_pred, labels=unique_classes, target_names=class_names, zero_division=0
        )
        print(report)
    else:
        print("No classes found in test set.")

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
    max_trigger_distance_len: int,
) -> None:
    """
    Save trained models and preprocessing artifacts to disk.

    Args:
        svm_model: Trained SVM model
        rf_model: Trained Random Forest model
        gb_model: Trained Gradient Boosting model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        max_hit_order_len: Maximum hit_order length for feature extraction
        max_chain_code_len: Maximum chain_code length for feature extraction
        max_palm_angles_len: Maximum palm_angles_flat length for feature extraction
        max_trigger_distance_len: Maximum trigger_distance length for feature extraction
    """
    print("\n" + "=" * 60)
    print("Saving models and artifacts...")
    print("=" * 60)

    # Save SVM model
    joblib.dump(svm_model, SVM_MODEL_PATH)
    print(f"Saved SVM model to: {SVM_MODEL_PATH}")

    # Save Random Forest model
    joblib.dump(rf_model, RF_MODEL_PATH)
    print(f"Saved Random Forest model to: {RF_MODEL_PATH}")

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to: {SCALER_PATH}")

    # Save label encoder
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")

    # Save Gradient Boosting model
    joblib.dump(gb_model, GB_MODEL_PATH)
    print(f"Saved Gradient Boosting model to: {GB_MODEL_PATH}")

    # Save feature configuration
    feature_config = {
        "max_hit_order_len": max_hit_order_len,
        "max_chain_code_len": max_chain_code_len,
        "max_palm_angles_len": max_palm_angles_len,
        "max_trigger_distance_len": max_trigger_distance_len,
    }
    joblib.dump(feature_config, FEATURE_CONFIG_PATH)
    print(f"Saved feature config to: {FEATURE_CONFIG_PATH}")

    print("\nAll artifacts saved successfully!")


def print_class_distribution(labels_list: List[str]) -> None:
    """
    Print the distribution of classes in the dataset.

    Args:
        labels_list: List of string labels
    """
    from collections import Counter

    label_counts = Counter(labels_list)
    print("\n" + "=" * 60)
    print("Class Distribution:")
    print("=" * 60)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples")
    print(f"\nTotal: {len(labels_list)} samples")


def main(dataset_paths: List[str] = None) -> None:
    """
    Main training pipeline.

    Args:
        dataset_paths: Optional list of dataset file paths.
                       If None, uses DEFAULT_DATASET_PATHS.
    """
    if dataset_paths is None:
        dataset_paths = DEFAULT_DATASET_PATHS

    print("=" * 60)
    print("ASL Sign Classification Training Pipeline")
    print("=" * 60)

    # Step 1: Load datasets
    print("\nStep 1: Loading datasets...")
    features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len = load_datasets(dataset_paths)

    # Print class distribution
    print_class_distribution(labels_list)

    # Step 2: Prepare features
    print("\nStep 2: Preparing features...")
    X_scaled, y_encoded, scaler, label_encoder = prepare_features(
        features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len
    )

    # Step 3: Split data
    print("\nStep 3: Splitting data into training and test sets...")
    
    # Check if stratified splitting is possible (need at least 2 samples per class)
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

    # Step 4: Train models
    svm_model = train_svm(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    # Step 5: Evaluate models
    svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM", label_encoder)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", label_encoder)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting", label_encoder)

    # Step 6: Compare models
    print("\n" + "=" * 60)
    print("Model Comparison:")
    print("=" * 60)
    print(f"\nSVM Accuracy:  {svm_metrics['accuracy']:.4f}")
    print(f"RF Accuracy:   {rf_metrics['accuracy']:.4f}")
    print(f"GB Accuracy:   {gb_metrics['accuracy']:.4f}")
    print(f"\nSVM F1-score:  {svm_metrics['f1_score']:.4f}")
    print(f"RF F1-score:   {rf_metrics['f1_score']:.4f}")
    print(f"GB F1-score:   {gb_metrics['f1_score']:.4f}")

    # User specified SVM as best model
    print(f"\n[+] Best performing model: SVM (as specified)")
    best_model = "SVM"

    # Find max trigger_distance length for saving
    max_trigger_distance_len = 0
    for feat_dict in features_list:
        trigger_distance = feat_dict.get("trigger_distance", [])
        max_trigger_distance_len = max(max_trigger_distance_len, len(trigger_distance))

    # Step 7: Save artifacts
    save_artifacts(svm_model, rf_model, gb_model, scaler, label_encoder, max_hit_order_len, max_chain_code_len, max_palm_angles_len, max_trigger_distance_len)

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)
    print(f"\nBest model: {best_model}")
    print(f"\nSaved files:")
    print(f"  - {SVM_MODEL_PATH}")
    print(f"  - {RF_MODEL_PATH}")
    print(f"  - {GB_MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    import sys

    # Allow custom dataset paths via command line arguments
    if len(sys.argv) > 1:
        dataset_paths = sys.argv[1:]
        main(dataset_paths)
    else:
        main()
