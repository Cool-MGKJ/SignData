# Training Pipeline Updates Documentation

## Overview

This document details all updates made to the ASL Sign Recognition training pipeline, including dataset format changes, feature engineering modifications, model configuration, and inference updates.

**Last Updated:** Based on training with `data/hand_sign_singlehand.json`

---

## Table of Contents

1. [Dataset Format](#dataset-format)
2. [Feature Engineering](#feature-engineering)
3. [Training Pipeline](#training-pipeline)
4. [Model Configuration](#model-configuration)
5. [Inference Updates](#inference-updates)
6. [Feature Configuration](#feature-configuration)
7. [Key Changes Summary](#key-changes-summary)

---

## Dataset Format

### Current Dataset Structure

The training pipeline now processes datasets in **array format** where each element contains metadata and samples:

```json
[
    {
        "metadata": {
            "total_samples": 107
        },
        "samples": [
            {
                "id": 1,
                "label": "Please",
                "points_left": [...],
                "points_right": [
                    {
                        "index": 0,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0
                    },
                    ...
                ],
                "hit_order": [1, 5, 12, ...],
                "chain_code": [0, 2, 4, ...],
                "palm_angles_left": [[pitch, yaw, roll], ...],
                "palm_angles_right": [[pitch, yaw, roll], ...],
                "trigger_distance_left": [...],
                "trigger_distance_right": [...]
            }
        ]
    }
]
```

### Dataset Processing Rules

1. **Hand Selection**: Uses only the **detected hand** (left OR right, whichever is present)
   - If both hands are present, uses the first non-empty hand
   - Single hand = 63 features (21 landmarks × 3 coordinates)

2. **Point Format**: Uses **numbered format directly** from `points_left` or `points_right`
   - Each point has `index`, `x`, `y`, `z` fields
   - Converted to flat array: `[x0, y0, z0, x1, y1, z1, ...]`

3. **Palm Angles**: Used as-is in `[pitch, yaw, roll]` format
   - Flattened from both hands: `[pitch1, yaw1, roll1, pitch2, yaw2, roll2, ...]`
   - Each entry is a 3-element array

4. **Excluded Features**: 
   - `trigger_distance_left` and `trigger_distance_right` are **excluded** from training

---

## Feature Engineering

### Feature Components

The final feature vector consists of **4 components**:

1. **Hand Landmarks (Points)**: 63 features
   - 21 landmarks × 3 coordinates (x, y, z)
   - Single hand only
   - Normalized using wrist-centered, rotation-aligned normalization

2. **Hit Order (Voxel Path)**: Variable length, padded to `max_hit_order_len`
   - Sequence of voxel indices hit during 3D grid tracking
   - Represents spatial trajectory of hand movement
   - Current max length: **20**

3. **Chain Code**: Variable length, padded to `max_chain_code_len`
   - 3D directional trajectory encoding
   - Represents movement direction between voxels
   - Current max length: **17**

4. **Palm Angles**: Variable length, padded to `max_palm_angles_len`
   - Flattened palm orientation angles: `[pitch, yaw, roll]` for each entry
   - Combined from both hands (if available)
   - Current max length: **117**

### Feature Vector Structure

```
Total Features = 63 + max_hit_order_len + max_chain_code_len + max_palm_angles_len
                = 63 + 20 + 17 + 117
                = 217 features
```

**Feature Order:**
```
[points_flat (63), hit_order_padded (20), chain_code_padded (17), palm_angles_padded (117)]
```

### Padding Strategy

- **Hit Order**: Padded with `0` values
- **Chain Code**: Padded with `0` values
- **Palm Angles**: Padded with `0.0` values
- **Points**: Padded/truncated to exactly 63 features (single hand)

---

## Training Pipeline

### Pipeline Steps

1. **Load Datasets**
   - Process array-formatted JSON files
   - Extract features: `points_flat`, `hit_order`, `chain_code`, `palm_angles_flat`
   - Track maximum lengths for each variable-length feature
   - Exclude `trigger_distance` features

2. **Prepare Features**
   - Pad all variable-length features to their maximum lengths
   - Combine into single feature vector (217 dimensions)
   - Encode string labels to integers using `LabelEncoder`
   - Scale features using `StandardScaler` (zero mean, unit variance)

3. **Split Data**
   - 80% training, 20% test split
   - Stratified split (if all classes have ≥2 samples)
   - Random state: 42 (for reproducibility)

4. **Train Model**
   - Currently trains **SVM only** (Support Vector Machine)
   - RBF kernel with `C=1.0`, `gamma='scale'`
   - Probability estimates enabled for confidence scores

5. **Evaluate Model**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - Classification report per class

6. **Save Artifacts**
   - Trained SVM model: `models/asl_svm_model.pkl`
   - Feature scaler: `models/scaler.pkl`
   - Label encoder: `models/label_encoder.pkl`
   - Feature config: `models/feature_config.pkl`

### Training Code Location

- **File**: `train_model.py`
- **Main function**: `main(dataset_paths)`
- **Usage**: 
  ```bash
  python train_model.py [dataset_path1] [dataset_path2] ...
  ```

---

## Model Configuration

### Current Model: SVM (Support Vector Machine)

**Hyperparameters:**
- **Kernel**: RBF (Radial Basis Function)
- **C**: 1.0 (regularization parameter)
- **Gamma**: 'scale' (automatic scaling)
- **Probability**: True (enables `predict_proba()` for confidence scores)
- **Random State**: 42

**Why SVM?**
- Effective for high-dimensional feature spaces
- Good generalization with limited data
- Provides probability estimates for confidence scoring
- Fast inference suitable for real-time applications

### Previously Tested Models

The pipeline previously supported:
- **Random Forest**: 100 estimators, unlimited depth
- **Gradient Boosting**: 200 estimators, learning rate 0.05

These were removed to focus on SVM for production use.

---

## Inference Updates

### Changes to `inference.py`

The inference module has been updated to match the training pipeline exactly:

#### 1. Single Hand Detection
- **Before**: Combined both hands (126 features)
- **After**: Uses only first detected hand (63 features)
- Matches training pipeline requirement

#### 2. Feature Extraction
- **Removed**: `trigger_distance` from feature extraction
- **Kept**: `points_flat`, `hit_order`, `chain_code`, `palm_angles`
- **Feature count**: 217 features (matches training)

#### 3. Palm Angles Format
- **Training format**: `[pitch, yaw, roll]` (as stored in dataset)
- **Live format**: `(yaw, pitch, roll)` (from `face_grid_3d.py`)
- **Conversion**: Reorders live angles to match training format

#### 4. Feature Configuration Loading
- Loads `max_hit_order_len`, `max_chain_code_len`, `max_palm_angles_len` from `feature_config.pkl`
- Uses these values for padding during inference
- Ensures feature dimensions match training exactly

#### 5. Auto-Reset Mechanism
- Resets grid tracking 1 second after first sign detection
- Confidence threshold: ≥ 0.5
- Clears accumulated tracking data for next sign

### Inference Workflow

```
1. Capture frame from webcam
2. Detect hand landmarks (MediaPipe)
3. Normalize landmarks (wrist-centered, rotation-aligned)
4. Extract features:
   - points_flat (63) from first detected hand
   - hit_order (padded to 20)
   - chain_code (padded to 17)
   - palm_angles (padded to 117, reordered to [pitch, yaw, roll])
5. Scale features using saved scaler
6. Predict label using SVM model
7. Display prediction and confidence
8. Auto-reset after 1 second if confidence ≥ 0.5
```

---

## Feature Configuration

### Current Configuration

Stored in `models/feature_config.pkl`:

```python
{
    "max_hit_order_len": 20,
    "max_chain_code_len": 17,
    "max_palm_angles_len": 117
}
```

### Feature Dimensions

| Feature Component | Dimensions | Description |
|------------------|------------|-------------|
| Points (landmarks) | 63 | Single hand, 21 landmarks × 3 coords |
| Hit Order | 20 | Voxel indices (padded) |
| Chain Code | 17 | Direction codes (padded) |
| Palm Angles | 117 | Flattened angles (padded) |
| **Total** | **217** | Combined feature vector |

### Configuration Usage

- **Training**: Determines padding lengths during feature preparation
- **Inference**: Ensures feature dimensions match training exactly
- **Both**: Must use same configuration for consistency

---

## Key Changes Summary

### From Previous Version

1. **Single Hand Only**
   - Changed from 126 features (both hands) to 63 features (single hand)
   - Uses first detected hand only

2. **Excluded Trigger Distance**
   - Removed `trigger_distance_left` and `trigger_distance_right` from features
   - Not included in training or inference

3. **Palm Angles Format**
   - Training uses `[pitch, yaw, roll]` format (as stored)
   - Inference converts from `(yaw, pitch, roll)` to match training

4. **Model Selection**
   - Trains only SVM (removed Random Forest and Gradient Boosting)
   - Focused on single best-performing model

5. **Feature Configuration**
   - Saves max lengths to `feature_config.pkl`
   - Ensures training and inference use same dimensions

6. **Dataset Format**
   - Supports array-formatted JSON files
   - Processes each array element separately
   - Uses numbered point format directly

### Breaking Changes

⚠️ **Important**: Models trained with the previous pipeline (126 features, with trigger_distance) are **not compatible** with the current inference code.

To use the current inference:
1. Retrain the model with the new dataset format
2. Ensure `feature_config.pkl` matches the new configuration
3. Verify feature dimensions: 217 total features

---

## File Locations

### Training Files
- `train_model.py` - Main training pipeline
- `models/asl_svm_model.pkl` - Trained SVM model
- `models/scaler.pkl` - Feature scaler
- `models/label_encoder.pkl` - Label encoder
- `models/feature_config.pkl` - Feature configuration

### Inference Files
- `inference.py` - Real-time inference interface
- `capture.py` - Webcam and MediaPipe integration
- `normalization.py` - Hand landmark normalization
- `face_grid_3d.py` - 3D grid tracking and feature extraction

### Dataset Files
- `data/hand_sign_singlehand.json` - Current training dataset
- `data/asl_dataset.json` - Previous dataset (legacy)
- `data/merged_asl_dataset.json` - Merged dataset (legacy)

---

## Usage Examples

### Training

```bash
# Train with default dataset
python train_model.py

# Train with custom dataset
python train_model.py "data/hand_sign_singlehand.json"

# Train with multiple datasets
python train_model.py "data/dataset1.json" "data/dataset2.json"
```

### Inference

```bash
# Run real-time inference
python inference.py
```

### Verify Feature Configuration

```python
import joblib

# Load and verify feature config
config = joblib.load('models/feature_config.pkl')
print(f"Hit order max length: {config['max_hit_order_len']}")
print(f"Chain code max length: {config['max_chain_code_len']}")
print(f"Palm angles max length: {config['max_palm_angles_len']}")

# Calculate total features
total = 63 + config['max_hit_order_len'] + config['max_chain_code_len'] + config['max_palm_angles_len']
print(f"Total features: {total}")
```

---

## Troubleshooting

### Feature Dimension Mismatch

**Error**: `ValueError: X has X features, but SVC is expecting Y features`

**Solution**:
1. Verify `feature_config.pkl` matches training configuration
2. Ensure inference uses same padding lengths
3. Check that single hand (63 features) is used, not both hands

### Model Not Loading

**Error**: `Model load error: invalid load key`

**Solution**:
1. Ensure models were saved with `joblib.dump()`
2. Load with `joblib.load()` (not `pickle.load()`)
3. Verify model files exist in `models/` directory

### Palm Angles Format Mismatch

**Issue**: Predictions are incorrect, suspect palm angles

**Solution**:
1. Verify palm angles are reordered: `(yaw, pitch, roll)` → `[pitch, yaw, roll]`
2. Check that angles are flattened correctly
3. Ensure padding matches `max_palm_angles_len`

---

## Future Enhancements

Potential improvements for the training pipeline:

1. **Hyperparameter Tuning**: Grid search or random search for optimal SVM parameters
2. **Feature Selection**: Identify most important features for dimensionality reduction
3. **Data Augmentation**: Generate synthetic samples to improve generalization
4. **Cross-Validation**: K-fold cross-validation for more robust evaluation
5. **Model Comparison**: Re-enable Random Forest and Gradient Boosting for comparison
6. **Deep Learning**: Consider CNN or LSTM for sequence-based recognition

---

## References

- **Training Script**: `train_model.py`
- **Inference Script**: `inference.py`
- **Feature Extraction**: `face_grid_3d.py`
- **Normalization**: `normalization.py`
- **Dataset Format**: `data/hand_sign_singlehand.json`

---

**Document Version**: 1.0  
**Last Training Date**: Based on `data/hand_sign_singlehand.json`  
**Model**: SVM (Support Vector Machine)  
**Total Features**: 217
