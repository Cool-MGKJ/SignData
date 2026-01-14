# SignData Project - Chronological Lifecycle Audit

## Project Overview
**Name**: SignData / ASL Dataset Collection Tool  
**Purpose**: Real-time ASL sign data collection using hand tracking, face-relative 3D spatial tracking, and gesture validation

---

## Phase 1: Initial Foundation (Pre-Conversation History)

### Core Architecture Established
- **Basic Components**:
  - `main.py`: Application entry point with tkinter UI
  - `capture.py`: MediaPipe hand detection and webcam handling
  - `normalization.py`: Initial normalization functions
  - `dataset_io.py`: Dataset storage (JSON/CSV export)
  - `ui.py`: Desktop GUI interface
  - `face_grid_3d.py`: Face-centered 3D grid tracking system

### Initial Features
- Real-time hand landmark detection (MediaPipe Hands, 21 points per hand)
- Basic landmark normalization
- Simple dataset collection and export
- Face-centered voxel grid (8×10×2 = **160 voxels**) with 2-layer depth system
  - Layer 0 (z_idx=0, voxels 0–79): Inner layer at face (green when triggered)
  - Layer 1 (z_idx=1, voxels 80–159): Outer layer at camera (red when triggered)
- Hit tracking during capture sessions

---

## Phase 2: Depth Estimation Evolution

### Initial Depth Implementation
- **MediaPipe z-depth**: Initially using MediaPipe's relative z-depth values
- **2D Grid System**: Early version may have been 2D only

### MiDaS Integration (Deprecated)
- **Feature**: Integration of MiDaS depth estimation model
- **Purpose**: Provide absolute depth measurements
- **Status**: **REMOVED** - All MiDaS code eliminated from codebase
- **Evidence**: README explicitly states "Uses MediaPipe's built-in relative z-depth" and references to MiDaS removed

### Current Depth System
- **MediaPipe z-depth exclusively**: All depth calculations use MediaPipe's relative z-coordinates
- **Relative depth**: Negative z = closer to camera, Positive z = farther from camera
- **Normalized range**: Z values normalized to 0-1 range for grid calculations

---

## Phase 3: Normalization Pipeline Development

### Early Normalization (`normalize_landmarks_to_3d_space`)
- **Method**: Wrist-based or centroid-based normalization
- **Options**:
  - `preserve_depth=True`: 2D palm scaling (x,y only) to preserve MediaPipe z-depth structure
  - `preserve_depth=False`: Legacy centroid-based normalization (deprecated)
- **Scaling**: Based on wrist-to-middle-finger-MCP distance
- **Status**: Still exists but legacy method deprecated

### Offline Batch Normalization (`normalize_dataset.py`)
- **Purpose**: Process existing JSON dataset files offline
- **Pipeline** (4-step):
  1. Wrist centering
  2. **2D palm scaling** (x,y only) - preserves MediaPipe z-depth structure
  3. Rotation alignment (wrist-to-middle-MCP → +Y axis)
  4. **Z-axis smoothing** (multiply z by 0.8 factor to reduce noise)
- **Data Format**: Dictionary format `{'index', 'x', 'y', 'z'}`
- **Output**: Normalized dataset files in `normalized_dataset/` folder

### Standard Hand Normalization (`normalize_hand_data`)
- **Purpose**: Live real-time normalization during capture
- **Pipeline** (3-step):
  1. Wrist centering (wrist → origin 0,0,0)
  2. **3D Euclidean scaling** (wrist-to-middle-MCP 3D distance)
  3. Rotation alignment (wrist-to-middle-MCP → +Y axis)
- **Note**: No z-axis smoothing step (unlike batch version)
- **Data Format**: Tuple format `(x, y, z)`
- **Status**: Current method used in live pipeline (when saving samples)

### Normalization Strategy Evolution
1. **Initially**: Normalization applied immediately after MediaPipe detection
2. **Problem Discovered**: Trigger point disappeared (coordinate system mismatch)
3. **Solution**: Raw MediaPipe coordinates for grid operations, normalization only when saving

---

## Phase 4: Depth Zone & Layer Triggering System

### Initial Z-Matching System
- **Method**: Hardcoded z-value ranges for layer selection
- **Issue**: Layer 1 not triggering correctly
- **Z-matching tolerance**: Adjustable (default: 0.25)

### Zone-Based Layer Switching (Hardcoded)
- **Zone 0 (Far/Near Face)**: z ∈ [0.350, 0.500] → Layer 0 (voxels 0-79)
- **Zone 1 (Near/Camera)**: z ∈ [0.100, 0.350] → Layer 1 (voxels 80-159)
- **Problem**: Distance-dependent, not adaptive to user size/distance

### Dynamic Depth Scaling System (Current)
- **Reference Length**: Body-relative measurement
  - Primary: Shoulder-to-shoulder distance (3D Euclidean)
  - Fallback: Wrist-to-elbow distance
  - Purpose: Make system distance-invariant
- **Face Z Reference**: Nose tip Z-value as zero plane
- **Relative Z Calculation**: `Relative_Z = (Raw_Z - Face_Z_Reference) / Reference_Length`
- **Layer Triggering Algorithm** (Formalized in Phase 9):
  ```
  if relative_z < 0:
      matching_layer_idx = 1  # Layer 1 (outer, z_idx=1, near camera) - RED
  else:
      matching_layer_idx = 0  # Layer 0 (inner, z_idx=0, near face) - GREEN
  ```
- **Zone Definitions** (Face-First Hierarchy):
  - **Layer 0 (Inner/Face)** (z_idx=0, voxels 0–79): `Relative_Z >= 0` (hand at/approaching face) - **green when triggered**
  - **Layer 1 (Outer/Camera)** (z_idx=1, voxels 80–159): `Relative_Z < 0` (hand extended toward camera) - **red when triggered**
- **Implementation**: MediaPipe Pose integration for shoulder/elbow detection
- **Status**: Current system, adaptive to user body size
- **Gesture Flow**: Hand enters Layer 1 first (outer), then Layer 0 (inner) as it approaches face
- **Key Design Properties**:
  - **Distance-invariant**: Works regardless of camera distance
  - **Adaptive**: Scales based on body size automatically
  - **Single threshold**: Predictable behavior with no fallback zones
  - **Reliable**: Based on body metrics, not hardcoded distances

### Layer Triggering Inversion
- **Initial Behavior**: Layer 1 triggered when hand moved away from camera
- **Desired Behavior**: Layer 1 triggers when hand moves closer to camera
- **Solution**: Inverted layer matching logic + relative Z zone system

---

## Phase 5: Motion Signature & Chain Code System

### Chain Code Implementation
- **Purpose**: Capture 3D trajectory of hand movement for gesture validation
- **Method**: 26-directional chain code (3D unit vectors)
  - 6 face directions (orthogonal): ±x, ±y, ±z
  - 12 edge directions (diagonal on one plane)
  - 8 corner directions (diagonal in all 3 axes)
- **Generation**: During voxel hit tracking, records direction from last voxel to current
- **Jitter Filtering**: Threshold (0.02) to filter small movements
- **Storage**: List of direction indices (0-25) appended to dataset

### Validation System
- **Method**: Levenshtein distance (SequenceMatcher) for chain code comparison
- **Function**: `validate_chain_code()` - compares current chain to target signature
- **Similarity Threshold**: Default 0.8 (80% similarity)
- **Combined Validation**: `validate_gesture()` - checks both layer sequence and chain code

### Capture Integration
- **Flag System**: `is_capturing_chain` boolean flag
- **Start/Stop**: Chain code capture only during active capture session
- **Storage**: Chain code appended to dataset samples as `chain_code` field

---

## Phase 6: Visualization & UI Enhancements

### Grid Visualization Evolution
- **Initial**: Basic grid point display (gray for unhit, green for hit)
- **Layer Differentiation**:
  - Layer 1 hits: Bright red `(0, 0, 255)` with radius 8
  - Layer 0 hits: Green (standard)
- **Path Visualization**: Magenta path lines (thickness reduced from 3→1, radius 3→2)

### Trigger Point Display
- **Visualization**: Large red circle with white outline
- **Label**: "TRIGGER" text next to point
- **Coordinate Display**: Trigger Z-value displayed on frame
- **Issue**: Disappeared temporarily due to coordinate system mismatch (fixed)

### Chain Code Display
- **Real-time Display**: Current chain code shown at bottom of frame
- **Format**: Comma-separated direction indices: `[0, 2, 5, ...]`

### Camera Feed
- **Mirror Effect**: Horizontal flip (`cv2.flip(frame, 1)`) for natural interaction

---

## Phase 7: Data Storage Evolution

### Initial Dataset Format
- **Basic Fields**: id, label, hand, num_points, points, timestamp
- **Points Format**: Flattened array `[x1, y1, z1, x2, y2, z2, ...]`

### Enhanced Dataset Format
- **Numbered Points**: `points` array with `{"index": 0, "x": ..., "y": ..., "z": ...}` format
- **Backward Compatibility**: `points_flat` maintained for older tools
- **Hit Order**: Ordered list of voxel indices visited during capture
- **Chain Code**: Sequence of direction indices (0-25) for 3D trajectory

### Export Formats
- **JSON**: Structured format with metadata and samples array
- **CSV**: Flattened format with columns for each coordinate
- **Auto-save**: Samples automatically saved to `data/asl_dataset.json` after each save

---

## Phase 8: Standard Hand Normalization (Latest)

### Implementation Request
- **Goal**: Minimize intra-class spread for better PCA clustering
- **Requirements**:
  1. Translation: Wrist at (0, 0, 0)
  2. Scaling: Standard size (wrist-to-middle-MCP distance = 1.0)
  3. Orientation: Wrist-to-middle-MCP aligned with +Y axis
  4. StandardScaler integration for feature standardization

### Implementation Details
- **Function**: `normalize_hand_data()` in `normalization.py`
- **Method**: 3D Euclidean distance scaling (unlike batch version which uses 2D)
- **Rotation**: Rodrigues' rotation formula for alignment
- **Integration**: Applied only when saving samples (not in live pipeline)
- **StandardScaler**: Optional sklearn integration for feature standardization

### Current State
- **Live Pipeline**: Uses raw MediaPipe coordinates for grid operations
- **Saving**: Applies "Standard Hand" normalization before saving
- **Result**: Consistent representation for ML training, grid operations unaffected

---

## Phase 9: Documentation Standardization & Layer Triggering Formalization (Latest)

### Architecture Guide Creation
- **File**: `docs/ARCHITECTURE_GUIDE.md` (NEW)
- **Purpose**: Comprehensive reference for all future code updates
- **Contents**:
  - Core layer system definition (Face-First naming)
  - Depth layer spacing calculation formula
  - Complete layer triggering algorithm with 5-step process
  - Dynamic depth scaling details
  - Chain code system (26-directional)
  - Hit detection algorithm
  - Data structures overview
  - Maintenance guidelines and testing checklist
- **Key Requirement**: Must be referenced for all future updates

### Documentation Consistency Pass
- **Files Updated**:
  1. `PROJECT_DOCUMENTATION.md`: Added "Layer Triggering Logic" subsection
  2. `docs/ARCHITECTURE_GUIDE.md`: Created with comprehensive system documentation
  3. `docs/API_REFERENCE.md`: Added "Layer Triggering Logic" section
  4. `docs/developer_guide.md`: Updated with cross-references
  5. `PROJECT_LIFECYCLE_AUDIT.md`: Updated with recent changes (this file)

### Consistency Fixes Applied
- **Voxel Count**: Unified to 160 (8×10×2) across all docs
- **Depth Layers**: Corrected from 3 to 2 across all docs
- **Layer Naming**: Face-First convention (Layer 0 at face, Layer 1 at camera) consistently applied
- **Normalization**: Clarified as "live-only" (no offline processing)
- **Chain Code**: Documented as post-processing only (validation functions removed)
- **Trigger Threshold**: Clearly stated as `relative_z < 0 → Layer 1`, `>= 0 → Layer 0`
- **Cross-references**: Added between all documentation files

### Parameter Corrections
- **depth_layers**: Fixed from 3 to 2 in `face_grid_3d.py` constructor
  - Line 123: Changed default from `depth_layers: int = 3` to `depth_layers: int = 2`
  - Updated docstrings and comments throughout file

### Documentation Quality Improvements
- **Invariants Documented**: Critical design rules that must be preserved
- **Testing Checklist**: Added to ARCHITECTURE_GUIDE for validation
- **Extension Points**: Clearly marked in API_REFERENCE for future developers
- **Visual Diagrams**: Added to PROJECT_DOCUMENTATION and ARCHITECTURE_GUIDE
- **Examples**: Added to API_REFERENCE and PROJECT_DOCUMENTATION

### Impact Assessment
- **Code Changes**: Minimal (only `depth_layers` parameter correction)
- **Documentation Impact**: Major (all files now consistent and comprehensive)
- **Functionality Impact**: None (documentation-only)
- **Maintenance Impact**: Positive (architecture guide will reduce future bugs)

---

### Removed Components
1. **Chain Code Validation Functions**: `validate_chain_code()` and `validate_gesture()` removed
   - Reason: Chain code is now stored for post-processing clustering, not real-time validation
   - Replacement: External tools can compare chain codes using Levenshtein distance or ML models

2. **Offline Normalization Script**: `normalize_dataset.py` deleted
   - Reason: Live normalization is canonical; no offline batch processing needed
   - All data saved in normalized form at capture time

3. **Deprecated Methods**:
   - Centroid-based normalization: `preserve_depth=False` (legacy)
   - Hardcoded z-zones: Replaced by dynamic depth scaling
   - MiDaS depth estimation: Completely removed from codebase

### Code Divergence Issues (RESOLVED)
- **Normalization Methods**: Two implementations existed
  - ❌ `normalize_dataset.py`: 2D scaling + z-smoothing (DELETED)
  - ✅ `normalize_hand_data()`: 3D scaling, no z-smoothing (CANONICAL)
- **Status**: Unified to single live normalization method

---

## Current Architecture

### Core Modules (Phase 9 State)
1. **main.py**: Application coordinator, state management
2. **capture.py**: MediaPipe hand/face/pose detection, frame processing
3. **face_grid_3d.py**: 3D voxel grid (160 voxels, 2 layers), hit tracking, chain code generation, **dynamic depth scaling**
4. **normalization.py**: Live normalization (wrist-centering + 3D scaling + rotation)
5. **dataset_io.py**: Dataset storage and export (JSON/CSV)
6. **ui.py**: Tkinter GUI interface

### Documentation Files (NEW)
- **docs/ARCHITECTURE_GUIDE.md**: System design, algorithms, invariants, testing (REFERENCE FOR ALL UPDATES)
- **docs/API_REFERENCE.md**: API signatures, usage examples, layer triggering details
- **docs/developer_guide.md**: Codebase structure, extension points, troubleshooting
- **PROJECT_DOCUMENTATION.md**: High-level overview, data formats, layer triggering
- **PROJECT_LIFECYCLE_AUDIT.md**: This file — chronological evolution

### Key Dependencies
- MediaPipe (Hands, Face Mesh, Pose)
- OpenCV (cv2)
- NumPy
- Tkinter (GUI)
- Pillow (image processing)
- scikit-learn (optional, for StandardScaler)

### Data Flow
1. **Capture**: MediaPipe → Raw landmarks (MediaPipe coordinates)
2. **Grid Operations**: Raw landmarks → Hit tracking (dynamic depth scaling), chain code generation
3. **Saving**: Raw landmarks → Live normalization → "Standard Hand" format → Save to dataset
4. **Export**: Dataset → JSON/CSV formats

---

## Phase 10: Palm Angle Tracking Feature (Current)

### Objective
Extend FaceGrid3D with real-time palm orientation angle tracking to capture hand rotation and orientation changes during gestures.

### Implementation Details

#### Infrastructure Added to `face_grid_3d.py`
- **Tracking Variables** (initialized in `__init__`):
  - `self.palm_angle_threshold = 5.0` (degrees) - trigger threshold for recording angle changes
  - `self.palm_angles = {}` - dictionary mapping hand_type → list of all recorded angles
  - `self.last_palm_angle = {}` - dictionary mapping hand_type → last recorded angle
  - `self.palm_angle_change_history = {}` - dictionary mapping hand_type → list of (angle, delta) tuples
  - `self.angle_change_tolerance = 0.1` (degrees) - noise filter for floating-point precision

- **Reset Logic**:
  - Updated `reset_hit_tracking()` to clear all palm angle tracking dictionaries

#### New Methods in `FaceGrid3D` Class

1. **`calculate_palm_angle(hand_landmarks) → Optional[float]`**
   - **Purpose**: Compute palm orientation angle from hand landmarks
   - **Algorithm**:
     - Extract wrist (index 0) and MCP landmarks (indices 5, 9, 13, 17)
     - Calculate palm plane normal vector using cross product: `(wrist→middle) × (wrist→ring)`
     - Compute angle relative to reference world vector [0, 1, 0]
     - Return angle in degrees (0-180 range)
   - **Error Handling**: Returns None if < 21 landmarks or vectors too parallel
   - **Performance**: O(1) - simple 3D vector math

2. **`update_palm_angle(hand_type, hand_landmarks) → Optional[Dict]`**
   - **Purpose**: Track angle changes and record when exceeding threshold
   - **Algorithm**:
     1. Calculate current angle via `calculate_palm_angle()`
     2. Initialize tracking if first occurrence of hand_type
     3. Calculate angle change with wraparound handling (359° → 1° = 2° change)
     4. If change > 5°: append (angle, delta) to change_history; record in palm_angles
     5. Return metadata dict with recorded status and reasoning
   - **Threshold Logic**:
     - Change > 5.0°: **RECORDED** (added to history)
     - Change > 0.1° but ≤ 5.0°: **NOT RECORDED** (updates internal state only)
     - Change ≤ 0.1°: **IGNORED** (noise floor)
   - **Return Format**:
     ```python
     {
         "hand": "right",
         "angle_deg": 45.2,
         "angle_change_deg": 7.5,
         "recorded": True,
         "reason": "exceeded_threshold_5.0deg"
     }
     ```

3. **`get_palm_angles(hand_type=None) → Dict`**
   - **Purpose**: Retrieve recorded palm angles
   - **Returns**: Dict mapping hand_type → list of angles, or filtered by hand_type

4. **`get_palm_angle_change_history(hand_type=None) → Dict`**
   - **Purpose**: Retrieve angle change history for analysis
   - **Returns**: Dict mapping hand_type → list of (angle, change_degrees) tuples

#### Integration Point
- **Location**: `update_hit_tracking()` method, after chain code update
- **Code**:
  ```python
  # Track palm angle changes for this hand
  hand_type = hand_data.get('hand_type', 'unknown')
  if hand_type in ['left', 'right']:
      palm_angle_info = self.update_palm_angle(hand_type, landmarks)
      # Palm angle tracking is optional; uncomment for debug output
  ```
- **Trigger**: Called once per hand per frame after hit tracking completes

#### Data Flow
```
[update_hit_tracking] 
  ↓
[for each hand_data in hand_landmarks_list]
  ├─ [Hit tracking & chain code] 
  │  
  └─ [Palm angle tracking]
      ├─ extract hand_type
      └─ call update_palm_angle(hand_type, landmarks)
          ├─ calculate_palm_angle()
          ├─ compare with last_palm_angle
          ├─ if delta > threshold: record change
          └─ return metadata
```

#### Feature Characteristics
- **Temporal Continuity**: Palm angle tracked frame-by-frame; only changes > 5° recorded
- **Hand Isolation**: Left and right hands tracked independently
- **Noise Resilience**: 
  - 0.1° tolerance prevents micro-fluctuations from triggering records
  - Angle wraparound (359° ↔ 1°) handled correctly as 2° change
- **Optional Debug Output**: Can uncomment print statements for real-time angle monitoring
- **Storage Format**: Angles as floats (degrees); change history as (angle, delta) tuples

#### Use Cases
1. **Gesture Classification**: Distinguish rotating vs. non-rotating hand poses
2. **Orientation Tracking**: Monitor palm-up/palm-down transitions during signs
3. **Hand Morphology**: Capture orientation changes in complex hand shapes
4. **Motion Analysis**: Identify twisting/rotation components of gestures

#### Testing Strategy
- Verify angle calculation with known hand orientations (palm-up=0°, palm-down=180°)
- Confirm 5° threshold not too sensitive (no spurious recordings) or too loose (missing real changes)
- Validate angle wraparound (test near 0°/360° boundary)
- Check independent tracking of left/right hands
- Ensure palm angles included in exported dataset

#### Backward Compatibility
- ✓ Fully backward compatible (tracking runs in parallel, non-intrusive)
- ✓ No impact on existing hit tracking or chain code logic
- ✓ Optional data (can be ignored if not needed)
- ✓ Default 5° threshold can be tuned via `palm_angle_threshold` parameter

### Current Status
- ✓ Infrastructure initialized (tracking variables, reset logic)
- ✓ Palm angle calculation method added
- ✓ Angle change tracking method added
- ✓ Data retrieval methods added
- ✓ Integration into update_hit_tracking() complete
- ⏳ Dataset export integration (optional - for capturing palm angles in saved gestures)
- ⏳ Visualization overlay (optional - show current palm angle on frame)

---

## Summary Statistics

### Codebase Evolution
| Metric | Count | Notes |
|--------|-------|-------|
| Depth Systems Tried | 2 | MiDaS → MediaPipe-only |
| Normalization Methods | 2 | 2D palm (removed) → 3D Standard Hand (current) |
| Layer Systems | 2 | Hardcoded zones → Dynamic depth scaling |
| Major Features Added | 4 | Chain code, layer triggering, depth scaling, documentation |
| Major Features Removed | 3 | MiDaS, validate_chain_code(), offline normalization |
| Documentation Files | 5 | PROJECT_DOCUMENTATION, ARCHITECTURE_GUIDE, API_REFERENCE, developer_guide, LIFECYCLE_AUDIT |

### Current Capabilities
- ✅ Real-time hand tracking (21 landmarks per hand)
- ✅ Face-relative 3D spatial tracking (160-voxel grid, 2 layers)
- ✅ **Dynamic depth-adaptive layer switching** (distance-invariant)
- ✅ 3D motion signature (26-directional chain code)
- ✅ Live hand normalization (wrist-center + 3D scaling + rotation)
- ✅ Multi-format dataset export (JSON/CSV)
- ✅ Comprehensive documentation with architecture guide

### Recent Improvements (Phase 9)
- ✅ Fixed `depth_layers` parameter from 3 to 2
- ✅ Created ARCHITECTURE_GUIDE.md as reference for all updates
- ✅ Standardized all documentation for consistency
- ✅ Documented complete layer triggering algorithm
- ✅ Added testing checklists and extension points
- ✅ Formalized design invariants that must be preserved

---

*Audit Generated: Based on codebase analysis and conversation history*  
*Project Status: Active Development*
