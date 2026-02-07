"""
Performance Testing and Visualization for ASL Sign Classification Models.

This script evaluates all trained models (SVM, Random Forest, Gradient Boosting)
on the test dataset and generates visualizations for:
- Accuracy, Precision, Recall, F1 Score (bar charts)
- Confusion matrices for each model
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import joblib


# Constants
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

SVM_MODEL_PATH = MODELS_DIR / "asl_svm_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "asl_rf_model.pkl"
GB_MODEL_PATH = MODELS_DIR / "asl_gb_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.pkl"

DEFAULT_DATASET_PATHS = [
    "data/combined_hand_sign_data ANGLE.json",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_datasets(dataset_paths: List[str]) -> Tuple[List[dict], List[str], int, int, int]:
    """
    Load samples from multiple JSON dataset files.
    
    (Reused from train_model.py for consistency)
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
                samples = []
                for item in data:
                    if isinstance(item, dict) and "samples" in item:
                        samples.extend(item.get("samples", []))
                    elif isinstance(item, dict) and "label" in item:
                        samples.append(item)
            else:
                samples = data.get("samples", [])
            
            if not samples:
                print(f"Warning: No samples found in {dataset_path}")
                continue

            for sample in samples:
                label = sample.get("label")
                if not label:
                    continue

                # Extract points - handle both formats (same as train_model.py)
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
                    points_flat = sample.get("points_flat", [])
                    if not points_flat:
                        points = sample.get("points", [])
                        if points and isinstance(points[0], dict):
                            # Convert numbered format to flat format
                            points_flat = []
                            for point_dict in points:
                                points_flat.extend([point_dict["x"], point_dict["y"], point_dict["z"]])
                
                if not points_flat or len(points_flat) == 0:
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

                # Update max lengths
                max_hit_order_len = max(max_hit_order_len, len(hit_order))
                max_chain_code_len = max(max_chain_code_len, len(chain_code))
                max_palm_angles_len = max(max_palm_angles_len, len(palm_angles_flat))

                # Store features as dict
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for evaluation (using same scaler/encoder from training).
    """
    # Load feature config to get the actual lengths used during training
    max_trigger_distance_len = 0
    if FEATURE_CONFIG_PATH.exists():
        feature_config = joblib.load(FEATURE_CONFIG_PATH)
        max_hit_order_len = feature_config.get("max_hit_order_len", max_hit_order_len)
        max_chain_code_len = feature_config.get("max_chain_code_len", max_chain_code_len)
        max_palm_angles_len = feature_config.get("max_palm_angles_len", max_palm_angles_len)
        max_trigger_distance_len = feature_config.get("max_trigger_distance_len", 0)
        print(f"\nUsing feature config from training:")
        print(f"  - Hit order: {max_hit_order_len}")
        print(f"  - Chain code: {max_chain_code_len}")
        print(f"  - Palm angles: {max_palm_angles_len}")
        print(f"  - Trigger distance: {max_trigger_distance_len}")

    padded_features = []
    
    # Use 126 features for both hands (21 landmarks × 3 coords × 2 hands) - same as training
    POINTS_FEATURES = 126
    
    for feat_dict in features_list:
        points_flat = feat_dict["points_flat"]
        hit_order = feat_dict.get("hit_order", [])
        chain_code = feat_dict.get("chain_code", [])
        palm_angles_flat = feat_dict.get("palm_angles_flat", [])
        trigger_distance = feat_dict.get("trigger_distance", [])
        
        # Pad/truncate points_flat to fixed size (126 for both hands) - same as training
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
        
        # Combine: points_flat + hit_order + chain_code + palm_angles_flat + trigger_distance (same as training)
        combined = points_padded + hit_order_padded + chain_code_padded + palm_angles_padded + trigger_distance_padded
        padded_features.append(combined)
    
    # Convert to numpy array
    X = np.array(padded_features, dtype=np.float32)
    y = np.array(labels_list)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"  - Points (both hands): 126")
    print(f"  - Hit order: {max_hit_order_len}")
    print(f"  - Chain code: {max_chain_code_len}")
    print(f"  - Palm angles: {max_palm_angles_len}")
    print(f"  - Trigger distance: {max_trigger_distance_len}")
    print(f"  - Total features: {X.shape[1]}")

    return X, y


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str, label_encoder: LabelEncoder
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data and return metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}...")
    print(f"{'='*60}")

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

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "y_test": y_test,
        "y_pred": y_pred,
        "label_encoder": label_encoder
    }


def plot_metrics_comparison(results: List[Dict[str, Any]], output_path: str = "model_performance_metrics.png"):
    """
    Create bar chart comparing accuracy, precision, recall, and F1 score across models.
    """
    model_names = [r["model_name"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    
    # Prepare data for plotting
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(), color=colors[i], alpha=0.8)
    
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison: Accuracy, Precision, Recall, F1 Score", 
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for r in results:
        idx = model_names.index(r["model_name"])
        for i, metric in enumerate(metrics):
            value = r[metric]
            ax.text(idx + i * width, value + 0.02, f"{value:.3f}", 
                   ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nMetrics comparison saved to: {output_path}")
    plt.close()


def plot_confusion_matrices(results: List[Dict[str, Any]], output_dir: str = "."):
    """
    Create confusion matrices for each model.
    """
    for result in results:
        model_name = result["model_name"]
        y_test = result["y_test"]
        y_pred = result["y_pred"]
        label_encoder = result["label_encoder"]
        
        # Get unique classes present in test/predictions
        unique_classes = sorted(set(np.concatenate([y_test, y_pred])))
        class_names = [label_encoder.classes_[idx] for idx in unique_classes]
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        
        # Normalize confusion matrix for better visualization
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # Create figure with two subplots: raw counts and normalized
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, 
                   yticklabels=class_names, ax=ax1, cbar_kws={"label": "Count"})
        ax1.set_title(f"{model_name} - Confusion Matrix (Counts)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Predicted", fontsize=10)
        ax1.set_ylabel("Actual", fontsize=10)
        
        # Plot normalized
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                   yticklabels=class_names, ax=ax2, cbar_kws={"label": "Normalized"})
        ax2.set_title(f"{model_name} - Confusion Matrix (Normalized)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Predicted", fontsize=10)
        ax2.set_ylabel("Actual", fontsize=10)
        
        plt.tight_layout()
        output_path = Path(output_dir) / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {output_path}")
        plt.close()


def main():
    """Main performance testing pipeline."""
    print("=" * 60)
    print("ASL Sign Classification - Performance Testing")
    print("=" * 60)

    # Step 1: Load datasets
    print("\nStep 1: Loading datasets...")
    features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len = load_datasets(DEFAULT_DATASET_PATHS)
    
    # Step 2: Prepare features
    print("\nStep 2: Preparing features...")
    X, y = prepare_features(features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len)

    # Step 3: Load scaler and label encoder
    print("\nStep 3: Loading preprocessing artifacts...")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")
    
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    # Transform features using the saved scaler
    X_scaled = scaler.transform(X)
    
    # Filter out samples with labels not seen during training
    valid_indices = []
    valid_labels = []
    for i, label in enumerate(y):
        if label in label_encoder.classes_:
            valid_indices.append(i)
            valid_labels.append(label)
        else:
            print(f"Warning: Skipping sample with unseen label '{label}' (not in training set)")
    
    if not valid_indices:
        raise ValueError("No valid samples found. All labels are unseen in the trained model.")
    
    # Filter features and labels
    X_scaled = X_scaled[valid_indices]
    y = np.array(valid_labels)
    
    print(f"Using {len(valid_indices)}/{len(X)} samples with valid labels")
    
    # Encode labels using the saved label encoder
    y_encoded = label_encoder.transform(y)

    # Step 4: Split data (using same random state as training)
    print("\nStep 4: Splitting data (matching training split)...")
    from collections import Counter
    class_counts = Counter(y_encoded)
    min_samples_per_class = min(class_counts.values())
    
    if min_samples_per_class < 2:
        print(f"Warning: Some classes have only 1 sample. Using non-stratified split.")
        use_stratify = False
    else:
        use_stratify = True
    
    if len(X_scaled) < 10:
        print(f"Warning: Very small dataset ({len(X_scaled)} samples). Using all data for testing.")
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y_encoded, y_encoded
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y_encoded if use_stratify else None
        )
    
    print(f"Test set: {X_test.shape[0]} samples")

    # Step 5: Load and evaluate models
    print("\nStep 5: Loading and evaluating models...")
    model_paths = {
        "SVM": SVM_MODEL_PATH,
        "Random Forest": RF_MODEL_PATH,
        "Gradient Boosting": GB_MODEL_PATH,
    }
    
    results = []
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            print(f"Warning: Model not found: {model_path}. Skipping {model_name}.")
            continue
        
        try:
            model = joblib.load(model_path)
            result = evaluate_model(model, X_test, y_test, model_name, label_encoder)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    if not results:
        raise ValueError("No models were successfully evaluated.")

    # Step 6: Generate visualizations
    print("\nStep 6: Generating visualizations...")
    plot_metrics_comparison(results)
    plot_confusion_matrices(results)

    # Step 7: Print summary
    print("\n" + "=" * 60)
    print("Performance Testing Summary")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model_name']:<20} {r['accuracy']:<10.4f} {r['precision']:<10.4f} "
              f"{r['recall']:<10.4f} {r['f1_score']:<10.4f}")
    
    print("\nPerformance testing completed!")


if __name__ == "__main__":
    main()
