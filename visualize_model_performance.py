"""
Visualize Model Performance: Accuracy Bar Chart and Confusion Matrices

This script:
1. Loads all trained models (SVM, Random Forest, Gradient Boosting)
2. Evaluates each model on test data
3. Creates accuracy comparison bar chart
4. Generates confusion matrices for each model
5. Saves visualizations to files
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
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

# Dataset path
DEFAULT_DATASET_PATH = "data/hand_sign_singlehand.json"

# Output paths
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_datasets(dataset_paths: List[str]) -> Tuple[List[dict], List[str], int, int, int]:
    """Load samples from multiple JSON dataset files."""
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
        
        # Handle array format - accumulate samples from all entries
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features for evaluation."""
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
    
    return X, y


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict:
    """Evaluate a model and return metrics."""
    try:
        # Check if model expects different feature dimensions
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            actual_features = X_test.shape[1]
            if expected_features != actual_features:
                raise ValueError(f"Feature dimension mismatch: model expects {expected_features}, got {actual_features}")
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get all classes for confusion matrix
        unique_classes = sorted(set(np.concatenate([y_test, y_pred])))
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        
        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "y_test": y_test,
            "y_pred": y_pred,
            "confusion_matrix": cm,
            "unique_classes": unique_classes
        }
    except ValueError as e:
        print(f"  Warning: {e}")
        return None


def plot_accuracy_comparison(results: List[Dict], output_path: Path):
    """Create bar chart comparing model accuracies."""
    model_names = [r["model_name"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]  # Convert to percentage
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#4CAF50', '#2196F3', '#FF9800'], alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison - Accuracy', fontsize=14, fontweight='bold')
    plt.ylim([0, 105])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line at 50% for reference
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50% Baseline')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy chart saved to: {output_path}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], model_name: str, output_path: Path):
    """Create confusion matrix heatmap."""
    plt.figure(figsize=(max(10, len(class_names) * 0.8), max(8, len(class_names) * 0.7)))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()


def main():
    """Main evaluation and visualization pipeline."""
    
    print("=" * 60)
    print("Model Performance Visualization")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\nStep 1: Loading dataset...")
    features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len = load_datasets([DEFAULT_DATASET_PATH])
    print(f"Loaded {len(features_list)} samples")
    
    # Load feature config to use same max lengths as training
    if FEATURE_CONFIG_PATH.exists():
        feature_config = joblib.load(FEATURE_CONFIG_PATH)
        max_hit_order_len = feature_config.get("max_hit_order_len", max_hit_order_len)
        max_chain_code_len = feature_config.get("max_chain_code_len", max_chain_code_len)
        max_palm_angles_len = feature_config.get("max_palm_angles_len", max_palm_angles_len)
        print(f"Using feature config: hit_order={max_hit_order_len}, chain_code={max_chain_code_len}, palm_angles={max_palm_angles_len}")
    
    # Step 2: Prepare features
    print("\nStep 2: Preparing features...")
    X, y = prepare_features(features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len)
    print(f"Feature matrix shape: {X.shape}")
    
    # Step 3: Load scaler and encoder
    print("\nStep 3: Loading scaler and label encoder...")
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    # Scale features
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    
    # Step 4: Split data (using same random state as training)
    print("\nStep 4: Splitting data...")
    from collections import Counter
    class_counts = Counter(y_encoded)
    min_samples_per_class = min(class_counts.values())
    
    use_stratify = min_samples_per_class >= 2
    
    if len(X_scaled) < 10:
        print("Warning: Very small dataset, using all data")
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y_encoded, y_encoded
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, 
            stratify=y_encoded if use_stratify else None
        )
    
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Load models
    print("\nStep 5: Loading models...")
    models = {}
    
    if SVM_MODEL_PATH.exists():
        models["SVM"] = joblib.load(SVM_MODEL_PATH)
        print("[+] SVM model loaded")
    else:
        print("[-] SVM model not found")
    
    if RF_MODEL_PATH.exists():
        models["Random Forest"] = joblib.load(RF_MODEL_PATH)
        print("[+] Random Forest model loaded")
    else:
        print("[-] Random Forest model not found")
    
    if GB_MODEL_PATH.exists():
        models["Gradient Boosting"] = joblib.load(GB_MODEL_PATH)
        print("[+] Gradient Boosting model loaded")
    else:
        print("[-] Gradient Boosting model not found")
    
    if not models:
        print("Error: No models found!")
        return
    
    # Step 6: Evaluate all models
    print("\nStep 6: Evaluating models...")
    results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(model, X_test, y_test, model_name)
        if result is not None:
            results.append(result)
            print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        else:
            print(f"  Skipping {model_name} due to feature dimension mismatch")
    
    if not results:
        print("\nError: No models could be evaluated successfully!")
        return
    
    # Step 7: Create visualizations
    print("\nStep 7: Creating visualizations...")
    
    # Accuracy bar chart
    accuracy_path = OUTPUT_DIR / "model_accuracy_comparison.png"
    plot_accuracy_comparison(results, accuracy_path)
    
    # Confusion matrices
    class_names = label_encoder.classes_
    
    for result in results:
        model_name = result["model_name"]
        cm = result["confusion_matrix"]
        unique_classes = result["unique_classes"]
        
        # Map numeric labels to class names
        cm_class_names = [class_names[idx] for idx in unique_classes]
        
        cm_path = OUTPUT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plot_confusion_matrix(cm, cm_class_names, model_name, cm_path)
    
    # Step 8: Print detailed metrics
    print("\n" + "=" * 60)
    print("Detailed Results")
    print("=" * 60)
    
    for result in results:
        model_name = result["model_name"]
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        
        # Classification report
        unique_classes = result["unique_classes"]
        class_names_subset = [class_names[idx] for idx in unique_classes]
        report = classification_report(result["y_test"], result["y_pred"], 
                                     target_names=class_names_subset, 
                                     labels=unique_classes,
                                     zero_division=0)
        print(f"\n  Classification Report:")
        print(report)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
