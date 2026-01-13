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
- Face-centered voxel grid (8×10×2 = 160 voxels) with 2-layer depth system
  - Layer 0 (z_idx=0): Inner layer at face (green when triggered)
  - Layer 1 (z_idx=1): Outer layer at camera (red when triggered)
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

### Zone-Based Layer Switching (Deprecated Hardcoded)
- **Old system**: Hardcoded z-value ranges
- **Problem**: Distance-dependent, not adaptive to user size/distance
- **Status**: REMOVED - replaced with dynamic depth scaling

### Dynamic Depth Scaling System (Current)
- **Reference Length**: Body-relative measurement
  - Primary: Shoulder-to-shoulder distance (3D Euclidean)
  - Fallback: Wrist-to-elbow distance
  - Purpose: Make system distance-invariant
- **Face Z Reference**: Nose tip Z-value as zero plane
- **Relative Z Calculation**: `Relative_Z = (Raw_Z - Face_Z_Reference) / Reference_Length`
- **Zone Definitions** (Face-First Hierarchy):
  - **Layer 0 (Inner/Face)** (z_idx=0, voxels 0-79): `Relative_Z >= 0` (hand at/approaching face) - **green when triggered**
  - **Layer 1 (Outer/Camera)** (z_idx=1, voxels 80-159): `Relative_Z < 0` (hand extended toward camera) - **red when triggered**
- **Implementation**: MediaPipe Pose integration for shoulder/elbow detection
- **Status**: Current system, adaptive to user body size
- **Gesture Flow**: Hand enters Layer 1 first (outer), then Layer 0 (inner) as it approaches face

### Layer Triggering & Naming Convention
- **Naming Strategy**: Face-First (z_idx alignment with layer names)
  - z_idx=0 → Layer 0 (at face)
  - z_idx=1 → Layer 1 (at camera)
- **Benefits**: Self-documenting code, reduces bugs, clearer indexing
- **Gesture Progression**: Hand moves through Layer 1 (outer) before Layer 0 (inner)
- **Color Coding**:
  - Layer 0 (z_idx=0, face): Green voxels (bright green when triggered)
  - Layer 1 (z_idx=1, camera): Red voxels (bright red when triggered)

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
- **Current Layer Coloring**:
  - Layer 0 (z_idx=0, face): Bright green when hit, dim green when unhit
  - Layer 1 (z_idx=1, camera): Bright red when hit, dim red when unhit
- **Path Visualization**: Magenta path lines (thickness 1, point radius 2)

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

## Technical Debt & Legacy Code

### Deprecated Methods
1. **Centroid-based normalization**: `preserve_depth=False` in `normalize_landmarks_to_3d_space()`
2. **Hardcoded z-zones**: Replaced by dynamic depth scaling
3. **MiDaS depth estimation**: Completely removed from codebase

### Code Divergence Issues
- **Normalization Methods**: Two different implementations
  - `normalize_dataset.py`: 2D scaling + z-smoothing
  - `normalize_hand_data()`: 3D scaling, no z-smoothing
- **Recommendation**: Unify normalization methods for consistency

---

## Current Architecture

### Core Modules
1. **main.py**: Application coordinator, state management
2. **capture.py**: MediaPipe hand detection, frame processing
3. **face_grid_3d.py**: 3D voxel grid, hit tracking, chain code generation, depth scaling
4. **normalization.py**: Normalization functions (multiple methods)
5. **normalize_dataset.py**: Offline batch normalization script
6. **dataset_io.py**: Dataset storage and export
7. **ui.py**: Tkinter GUI interface

### Key Dependencies
- MediaPipe (Hands, Face Mesh, Pose)
- OpenCV (cv2)
- NumPy
- Tkinter (GUI)
- scikit-learn (optional, for StandardScaler)
- Pillow (image processing)

### Data Flow
1. **Capture**: MediaPipe → Raw landmarks (MediaPipe coordinates)
2. **Grid Operations**: Raw landmarks → Hit tracking, chain code generation
3. **Saving**: Raw landmarks → Normalize → "Standard Hand" format → Save to dataset
4. **Export**: Dataset → JSON/CSV formats

---

## Summary Statistics

### Codebase Evolution
- **Depth Systems**: 2 (MiDaS → MediaPipe-only)
- **Normalization Methods**: 3 (centroid-based → 2D palm → 3D Standard Hand)
- **Layer Systems**: 2 (hardcoded zones → dynamic depth scaling)
- **Major Features Added**: Chain code, motion signatures, dynamic depth scaling
- **Major Features Removed**: MiDaS integration

### Current Capabilities
- Real-time hand tracking (21 landmarks per hand)
- Face-relative 3D spatial tracking (240 voxel grid)
- Dynamic depth-adaptive layer switching
- 3D motion signature (26-directional chain code)
- Standard Hand normalization for ML training
- Multi-format dataset export (JSON/CSV)

---

*Audit Generated: Based on codebase analysis and conversation history*  
*Project Status: Active Development*
