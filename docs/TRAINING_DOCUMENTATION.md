# ASL Sign Recognition Training Documentation

## Overview

This document provides comprehensive documentation of the machine learning training pipeline for ASL (American Sign Language) sign recognition, including model selection rationale, feature engineering, and training procedures.

**Current Model**: Support Vector Machine (SVM) with RBF kernel  
**Training Dataset**: `data/hand_sign_singlehand.json`  
**Total Features**: 217 dimensions  
**Model File**: `models/asl_svm_model.pkl`

---

## Table of Contents

1. [Training Pipeline Overview](#training-pipeline-overview)
2. [Model Selection: Why SVM?](#model-selection-why-svm)
3. [Random Forest: Why It Wasn't Used](#random-forest-why-it-wasnt-used)
4. [Gradient Boosting: Why It Wasn't Used](#gradient-boosting-why-it-wasnt-used)
5. [Feature Engineering](#feature-engineering)
6. [Training Process](#training-process)
7. [Model Configuration](#model-configuration)
8. [Current Implementation](#current-implementation)

---

## Training Pipeline Overview

### Pipeline Steps

The training pipeline follows these sequential steps:

1. **Dataset Loading** → Load JSON dataset files
2. **Feature Extraction** → Extract hand landmarks, voxel paths, chain codes, palm angles
3. **Feature Preparation** → Pad, combine, and normalize features
4. **Data Splitting** → Split into training (80%) and test (20%) sets
5. **Model Training** → Train SVM classifier
6. **Model Evaluation** → Evaluate on test set
7. **Artifact Saving** → Save model, scaler, encoder, and config

### Code Location

- **Training Script**: `train_model.py`
- **Main Function**: `main(dataset_paths)`
- **Usage**:
  ```bash
  python train_model.py [dataset_path1] [dataset_path2] ...
  ```

---

## Model Selection: Why SVM?

### Current Choice: Support Vector Machine (SVM)

The training pipeline **currently trains and uses only SVM** (Support Vector Machine with RBF kernel).

### SVM Advantages for ASL Recognition

1. **High-Dimensional Feature Space Handling**
   - Our feature vector has 217 dimensions
   - SVM with RBF kernel excels at finding complex decision boundaries in high-dimensional spaces
   - Works well even when number of features exceeds number of samples

2. **Small to Medium Dataset Performance**
   - ASL datasets are often limited in size (hundreds to thousands of samples)
   - SVM generalizes well with limited training data
   - Less prone to overfitting compared to complex tree-based models

3. **Probability Estimates**
   - SVM with `probability=True` provides confidence scores via `predict_proba()`
   - Essential for real-time inference where confidence thresholds matter
   - Enables filtering of low-confidence predictions

4. **Fast Inference**
   - Once trained, SVM prediction is fast (suitable for real-time applications)
   - Supports frame-by-frame prediction in live video streams
   - Minimal computational overhead in production

5. **Effective with Normalized Features**
   - Our features are preprocessed with `StandardScaler` (zero mean, unit variance)
   - SVM performs optimally with scaled features
   - RBF kernel is particularly sensitive to feature scaling

### SVM Configuration

```python
SVM Model Parameters:
- Kernel: RBF (Radial Basis Function)
- C: 1.0 (regularization parameter - controls trade-off between margin and error)
- Gamma: 'scale' (automatic scaling based on feature variance)
- Probability: True (enables predict_proba for confidence scores)
- Random State: 42 (for reproducibility)
```

**Code Reference**: `train_model.py`, lines 306-332

---

## Random Forest: Why It Wasn't Used

### What is Random Forest?

Random Forest is an ensemble method that constructs multiple decision trees during training and outputs the mode of predictions from individual trees.

### Random Forest Configuration (Available but Not Used)

```python
Random Forest Parameters:
- n_estimators: 100 (number of decision trees)
- max_depth: None (unlimited depth)
- random_state: 42
- n_jobs: -1 (use all CPU cores)
```

**Code Reference**: `train_model.py`, lines 335-369

### Why Random Forest Was Not Selected

1. **Overfitting Risk with High-Dimensional Features**
   - Random Forest can overfit with 217 features and limited samples
   - Each tree learns from a subset of features, but with many features, selection becomes less meaningful
   - Risk of memorizing training data rather than generalizing

2. **Less Effective with Small Datasets**
   - Tree-based models typically require more data to generalize well
   - Random Forest performs better with large, diverse datasets
   - Our ASL datasets are often small to medium-sized

3. **Feature Importance Concerns**
   - While Random Forest provides feature importance, with 217 features including padded sequences, interpretation becomes difficult
   - Many features are padded zeros, which can confuse importance calculations

4. **Slower Training Time**
   - Training 100 decision trees is computationally expensive
   - Not necessary when SVM achieves comparable or better results faster

5. **Inference Complexity**
   - Random Forest requires evaluating 100 trees per prediction
   - More complex model structure increases inference time
   - Less suitable for real-time applications

6. **Empirical Performance**
   - During development and testing, SVM consistently outperformed Random Forest on validation sets
   - Better accuracy and generalization on unseen data
   - More stable predictions across different sign variations

### Current Status

- **Training Function**: `train_random_forest()` exists in `train_model.py` (lines 335-369)
- **Usage**: **Not called** in the main training pipeline
- **Model Path**: `RF_MODEL_PATH` defined but not used for saving
- **Reason**: SVM was chosen as the production model after comparative evaluation

### Enabling Random Forest (For Comparison)

To train Random Forest for comparison purposes, you would modify `main()` in `train_model.py`:

```python
# In main() function, after SVM training:
rf_model = train_random_forest(X_train, y_train)
rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", label_encoder)
```

However, this is **not recommended** for production use as SVM has been validated as the better choice.

---

## Gradient Boosting: Why It Wasn't Used

### What is Gradient Boosting?

Gradient Boosting is an ensemble technique that builds models sequentially, where each new model corrects errors made by previous models.

### Gradient Boosting Configuration (Available but Not Used)

```python
Gradient Boosting Parameters:
- n_estimators: 200 (number of boosting stages)
- learning_rate: 0.05 (step size shrinkage)
- max_depth: 3 (maximum depth of individual trees)
- subsample: 0.9 (fraction of samples used for each tree)
- random_state: 42
```

**Code Reference**: `train_model.py`, lines 372-397

### Why Gradient Boosting Was Not Selected

1. **Extremely Slow Training**
   - Sequential training of 200 boosting stages is computationally intensive
   - Much slower than SVM for comparable performance
   - Not suitable for rapid iteration during development

2. **Hyperparameter Sensitivity**
   - Gradient Boosting requires careful tuning of learning_rate, n_estimators, max_depth
   - Finding optimal parameters requires extensive grid search
   - SVM with default 'scale' gamma performs well without extensive tuning

3. **Overfitting Risk**
   - Gradient Boosting can easily overfit with small datasets
   - Requires careful regularization via learning_rate and max_depth
   - More complex to tune correctly

4. **Not Necessary for This Problem**
   - SVM with RBF kernel already captures non-linear patterns effectively
   - Adding boosting complexity doesn't provide sufficient performance gains
   - Diminishing returns for the computational cost

5. **Memory and Storage**
   - Gradient Boosting models can be large (200 trees with feature information)
   - More memory-intensive during training and inference
   - Model files are larger on disk

6. **Real-Time Inference Concerns**
   - Evaluating 200 sequential trees per prediction is slower than SVM
   - Not optimal for frame-by-frame real-time sign recognition
   - Latency matters in live video applications

### Current Status

- **Training Function**: `train_gradient_boosting()` exists in `train_model.py` (lines 372-397)
- **Usage**: **Not called** in the main training pipeline
- **Model Path**: `GB_MODEL_PATH` defined but not used for saving
- **Reason**: SVM was chosen due to better balance of performance, speed, and simplicity

### Enabling Gradient Boosting (For Comparison)

To train Gradient Boosting for comparison purposes, you would modify `main()` in `train_model.py`:

```python
# In main() function, after SVM training:
gb_model = train_gradient_boosting(X_train, y_train)
gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting", label_encoder)
```

However, this is **not recommended** due to training time and lack of performance advantage over SVM.

---

## Feature Engineering

### Feature Components

The model uses **4 types of features** combined into a single 217-dimensional vector:

#### 1. Hand Landmarks (63 features)

- **Source**: MediaPipe hand landmarks (21 points × 3 coordinates)
- **Normalization**: Wrist-centered, rotation-aligned, scale-normalized
- **Format**: Flat array `[x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]`
- **Hand Selection**: Uses **only detected hand** (left OR right, first available)

#### 2. Hit Order / Voxel Path (20 features)

- **Source**: 3D voxel grid tracking (8×10×2 = 160 voxels)
- **Description**: Sequence of voxel indices hit during hand movement
- **Max Length**: 20 (padded with zeros if shorter)
- **Purpose**: Captures spatial trajectory of hand motion

#### 3. Chain Code (17 features)

- **Source**: 3D directional trajectory encoding
- **Description**: Direction codes between consecutive voxels
- **Max Length**: 17 (padded with zeros if shorter)
- **Purpose**: Encodes movement direction and velocity

#### 4. Palm Angles (117 features)

- **Source**: Palm orientation angles from face grid tracking
- **Format**: Flattened `[pitch, yaw, roll]` arrays from both hands
- **Max Length**: 117 (padded with zeros if shorter)
- **Purpose**: Captures hand orientation and rotation

### Feature Vector Structure

```
Total Features = 63 + 20 + 17 + 117 = 217 dimensions

Feature Vector Layout:
[hand_landmarks (63), hit_order (20), chain_code (17), palm_angles (117)]
```

### Feature Processing Pipeline

1. **Extraction**: Extract raw features from dataset JSON
2. **Padding**: Pad variable-length features to maximum lengths
3. **Combination**: Concatenate all feature components
4. **Scaling**: Apply `StandardScaler` (zero mean, unit variance)
5. **Training**: Feed scaled features to SVM

**Code Reference**: `train_model.py`, `load_datasets()` (lines 68-217) and `prepare_features()` (lines 220-303)

---

## Training Process

### Step-by-Step Training Procedure

#### Step 1: Load Datasets

```python
features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len = load_datasets(dataset_paths)
```

- Loads JSON dataset files (array format)
- Extracts features: `points_flat`, `hit_order`, `chain_code`, `palm_angles_flat`
- Tracks maximum lengths for variable-length features
- Returns feature dictionaries and labels

#### Step 2: Prepare Features

```python
X_scaled, y_encoded, scaler, label_encoder = prepare_features(
    features_list, labels_list, max_hit_order_len, max_chain_code_len, max_palm_angles_len
)
```

- Pads all variable-length features to maximum lengths
- Combines into single feature vector (217 dimensions)
- Encodes string labels to integers using `LabelEncoder`
- Scales features using `StandardScaler` (required for SVM)

#### Step 3: Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
```

- 80% training, 20% test split
- Stratified split (maintains class distribution)
- Random state 42 for reproducibility

#### Step 4: Train SVM Model

```python
svm_model = train_svm(X_train, y_train)
```

- Creates SVC with RBF kernel
- Sets `probability=True` for confidence scores
- Fits model to training data

#### Step 5: Evaluate Model

```python
svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM", label_encoder)
```

- Calculates accuracy, precision, recall, F1-score
- Generates confusion matrix
- Prints classification report per class

#### Step 6: Save Artifacts

```python
save_artifacts(svm_model, scaler, label_encoder, max_hit_order_len, max_chain_code_len, max_palm_angles_len)
```

Saves:
- `models/asl_svm_model.pkl` - Trained SVM model
- `models/scaler.pkl` - Feature scaler
- `models/label_encoder.pkl` - Label encoder
- `models/feature_config.pkl` - Feature configuration (max lengths)

**Code Reference**: `train_model.py`, `main()` function (lines 533-604)

---

## Model Configuration

### SVM Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kernel` | `'rbf'` | Radial Basis Function kernel for non-linear classification |
| `C` | `1.0` | Regularization parameter (higher = less regularization) |
| `gamma` | `'scale'` | Kernel coefficient (auto-scaled based on feature variance) |
| `probability` | `True` | Enable probability estimates via Platt scaling |
| `random_state` | `42` | Seed for reproducibility |

### Training Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `TEST_SIZE` | `0.2` | 20% of data reserved for testing |
| `RANDOM_STATE` | `42` | Random seed for reproducible splits |
| `stratify` | `True` | Maintain class distribution in train/test split |

### Feature Configuration

Stored in `models/feature_config.pkl`:

```python
{
    "max_hit_order_len": 20,
    "max_chain_code_len": 17,
    "max_palm_angles_len": 117
}
```

These values are determined during dataset loading and used for feature padding in both training and inference.

---

## Current Implementation

### What Is Actually Used

1. **Model**: SVM (Support Vector Machine with RBF kernel)
2. **Features**: 217-dimensional vector (63 + 20 + 17 + 117)
3. **Preprocessing**: StandardScaler (zero mean, unit variance)
4. **Label Encoding**: LabelEncoder (string labels → integers)
5. **Inference**: Real-time prediction via `inference.py`

### Training Functions Available (But Not Used)

The following training functions exist in `train_model.py` but are **not called** in the main pipeline:

1. **`train_random_forest()`** (lines 335-369)
   - Available but not used
   - Reason: SVM performs better for this use case

2. **`train_gradient_boosting()`** (lines 372-397)
   - Available but not used
   - Reason: Too slow, no performance advantage over SVM

### Saved Model Artifacts

After training, the following files are saved:

```
models/
├── asl_svm_model.pkl      # Trained SVM model (USED)
├── scaler.pkl             # Feature scaler (USED)
├── label_encoder.pkl      # Label encoder (USED)
└── feature_config.pkl     # Feature configuration (USED)
```

**Note**: `asl_rf_model.pkl` and `asl_gb_model.pkl` are **not created** because Random Forest and Gradient Boosting are not trained.

### Inference Pipeline

The inference module (`inference.py`) uses the trained SVM model:

```python
# Load model artifacts
model = joblib.load("models/asl_svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_config = joblib.load("models/feature_config.pkl")

# Extract and prepare features (matching training pipeline)
features = extract_features(points, hit_order, chain_code, palm_angles)

# Scale features
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)
confidence = model.predict_proba(features_scaled)[0].max()
```

---

## Summary

### Model Selection Decision

| Model | Status | Reason |
|-------|--------|--------|
| **SVM** | ✅ **USED** | Best balance of performance, speed, and generalization |
| **Random Forest** | ❌ Available but not used | Overfitting risk, slower inference, less effective with small datasets |
| **Gradient Boosting** | ❌ Available but not used | Too slow, hyperparameter sensitivity, no performance advantage |

### Key Takeaways

1. **SVM is the production model** - Trained, evaluated, and saved for real-time inference
2. **Random Forest and Gradient Boosting functions exist** - Can be enabled for comparison but not recommended
3. **Feature engineering is critical** - 217 features combining static pose, dynamic motion, and orientation
4. **Preprocessing is essential** - StandardScaler normalization required for SVM performance
5. **Single hand detection** - Uses only detected hand (63 features), not both hands

### Training Command

```bash
# Train with default dataset
python train_model.py

# Train with custom dataset
python train_model.py "data/hand_sign_singlehand.json"
```

### Verification

After training, verify the model:

```python
import joblib

# Load and inspect
model = joblib.load("models/asl_svm_model.pkl")
config = joblib.load("models/feature_config.pkl")

print(f"Model type: {type(model)}")
print(f"Feature config: {config}")
print(f"Total features: {63 + config['max_hit_order_len'] + config['max_chain_code_len'] + config['max_palm_angles_len']}")
```

Expected output:
```
Model type: <class 'sklearn.svm._classes.SVC'>
Feature config: {'max_hit_order_len': 20, 'max_chain_code_len': 17, 'max_palm_angles_len': 117}
Total features: 217
```

---

**Document Version**: 1.0  
**Last Updated**: Based on training with `data/hand_sign_singlehand.json`  
**Model**: SVM (Support Vector Machine)  
**Status**: Production-ready
