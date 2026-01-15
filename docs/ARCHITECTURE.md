# System Architecture

Complete overview of the SignData system design, components, and how they interact.

## System Overview

SignData is an ASL gesture recognition system that:
1. Captures hand landmarks in real-time using MediaPipe
2. Maps hands to a 3D face-centered voxel grid
3. Records spatial hits, movement trajectory, hand orientation, and distance
4. Exports labeled samples for machine learning

### Key Statistics
- **Grid Size:** 160 voxels (8 wide × 10 tall × 2 deep)
- **Layers:** 2 (Layer 0 near face, Layer 1 near camera)
- **Hand Landmarks:** 21 per hand
- **Output Dimensions:** Hand coordinates + spatial hits + chain code + angles + distance

---

## Component Architecture

```
┌─────────────────────────────────────────────────────┐
│                   main.py                            │
│            (Application Coordinator)                 │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┼──────────┬──────────┐
        │          │          │          │
        v          v          v          v
    capture.py  dataset_io.py ui.py   normalization.py
    (Camera)    (Storage)   (UI)      (Normalization)
        │
        v
   face_grid_3d.py
   (3D Grid Tracking)
        │
        ├─ MediaPipe Hand
        ├─ MediaPipe Face Mesh
        └─ MediaPipe Pose
```

### Component Responsibilities

| Component | Purpose |
|-----------|---------|
| **main.py** | Orchestrates capture, saving, and export |
| **capture.py** | Manages camera input and MediaPipe hand detection |
| **face_grid_3d.py** | Core 3D voxel grid tracking and visualization |
| **dataset_io.py** | In-memory storage and file export (JSON/CSV) |
| **ui.py** | Tkinter-based user interface |
| **normalization.py** | Hand landmark normalization and transformation |

---

## The 2-Layer Voxel Grid System

### Visual Layout

```
┌─────────────────────────────────────┐
│                CAMERA               │
├─────────────────────────────────────┤
│                                     │
│   [RED LAYER 1]  ← Voxels 80-159   │
│   Hand extended toward camera       │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  [GREEN LAYER 0]  ← Voxels 0-79    │
│  Hand at/toward face                │
│                                     │
├─────────────────────────────────────┤
│                FACE                 │
└─────────────────────────────────────┘
```

### Grid Dimensions

- **Width (X):** 8 voxels
- **Height (Y):** 10 voxels  
- **Depth (Z):** 2 layers
- **Total:** 8 × 10 × 2 = **160 voxels**

### Voxel Indexing

Voxels are indexed in order: [z, row, col]
```
Index = z_idx * (breadth * length) + y_idx * breadth + x_idx
```

**Examples:**
- Voxel(x=0, y=0, z=0) = Index 0
- Voxel(x=7, y=9, z=0) = Index 79 (last voxel in Layer 0)
- Voxel(x=0, y=0, z=1) = Index 80 (first voxel in Layer 1)
- Voxel(x=7, y=9, z=1) = Index 159 (last voxel in Layer 1)

---

## Face-Centered Coordinate System

### Origin and Axes

- **Origin:** Nose tip (MediaPipe Face Mesh landmark 4)
- **X-axis:** Left-right (0 = left, 1 = right)
- **Y-axis:** Up-down (0 = top, 1 = bottom)
- **Z-axis:** Forward-back (relative depth from face)

### Coordinate Range

All coordinates are **normalized to [0, 1]**:
- X ∈ [0, 1] represents full frame width
- Y ∈ [0, 1] represents full frame height
- Z ∈ [0, 1] represents depth relative to face

### Dynamic Positioning

The grid automatically:
1. **Centers** on the nose
2. **Rotates** with head yaw and pitch
3. **Scales** based on face size (eye distance)
4. **Updates** every frame to track head movement

---

## Head Tracking and Rotation

### Detecting Head Orientation

**Yaw (Left-Right Turn):**
- Calculated from eye positions
- Formula: `yaw = arctan2(right_eye.z - left_eye.z, right_eye.x - left_eye.x)`
- Range: ±40° (configurable)

**Pitch (Up-Down Nod):**
- Calculated from forehead to chin vector
- Formula: `pitch = arctan2(chin.z - forehead.z, chin.y - forehead.y)`
- Range: ±30° (configurable)

### Grid Rotation

The grid rotates with the head:
1. **Yaw Rotation** - Rotates around Y-axis
2. **Pitch Rotation** - Rotates around X-axis
3. **Applied** - To all voxel positions to keep grid aligned with face

This ensures the grid always "faces" the hand gestures correctly regardless of head orientation.

---

## Depth Scaling: Layer Triggering Algorithm

### The Problem
- Different people have different reach distances
- Hand depth varies dramatically based on gesture
- Need automatic, adaptive layer selection

### The Solution: Dynamic Depth Scaling

Convert raw hand Z-coordinate to relative Z:
```
relative_z = (hand_z - face_z_reference) / reference_length
```

Where:
- `hand_z` = Raw MediaPipe hand Z-coordinate
- `face_z_reference` = Nose tip Z (Face Mesh landmark 4)
- `reference_length` = Shoulder-to-shoulder distance (from Pose)

### Layer Selection

```
if relative_z >= 0:
    → Layer 0 (GREEN) - hand at/toward face
else:
    → Layer 1 (RED) - hand extended toward camera
```

### Algorithm Steps

1. **Get Reference Length** - Shoulder distance from Pose
2. **Get Face Z Reference** - Nose Z from Face Mesh
3. **Get Hand Z** - Raw hand Z from Hand landmarks
4. **Calculate relative_z** - Apply formula above
5. **Compare to 0** - Determine which layer
6. **Select Voxels** - Use only Layer 0 or Layer 1
7. **Find Nearest** - 2D search in selected layer
8. **Trigger** - If within hit radius

### Why This Works

- **Distance-invariant** - Accounts for individual differences
- **Adaptive** - Works with different hand speeds
- **Robust** - Handles partial hand visibility
- **Simple** - Single threshold comparison

---

## Visualization and Rendering

### 2D Projection

3D voxel centers are projected to 2D pixels:
```
pixel_x = voxel_x * frame_width
pixel_y = voxel_y * frame_height
```

### Rendering Order

Voxels are sorted by depth and rendered back-to-front:
- Farthest voxels (Layer 1) drawn first
- Nearest voxels (Layer 0) drawn last
- Creates 3D depth perception

### Color Scheme

| Layer | Color | Hit Status | Unhit Status |
|-------|-------|-----------|--------------|
| 0 (Face) | Green | Bright Green | Dim Green |
| 1 (Camera) | Red | Bright Red | Dim Red |

### Visual Features

- **Voxel Indices:** Overlaid numbers (0-159) for reference
- **Path Line:** Polyline showing order of voxel touches
- **Offset:** Slight horizontal offset per layer for clarity
- **Size:** Larger circles for hit voxels, smaller for unhit
- **Grid Frame:** Shows the overall 3D structure

---

## Data Collection

### Per-Gesture Data

Each captured gesture records:

1. **Normalized Hand Landmarks** (21 per hand)
   - 3D coordinates (x, y, z)
   - Wrist-centered normalization
   - Unit scale

2. **Hit Order** (sequence of voxel touches)
   - Example: `[31, 23, 103, 39, 47, ...]`
   - Shows path through voxel grid

3. **Chain Code** (directional trajectory)
   - 26-directional encoding (0-25)
   - Example: `[3, 4, 15, 2, 1, 8, ...]`
   - Captures 3D movement pattern

4. **Palm Angles** (hand orientation)
   - Yaw, pitch, roll in degrees
   - Separate for left and right hands
   - Recorded when change > 5°

5. **Distance Tracking** (hand-to-face distance)
   - Euclidean distance in normalized coordinates
   - Separate for left and right hands
   - Recorded when change > 0.05 units

---

## Data Pipeline

```
Capture Frame
    ↓
[MediaPipe Hand] → 21 landmarks per hand
[MediaPipe Face] → Face orientation & position
[MediaPipe Pose] → Shoulder distance (reference)
    ↓
Process Frame
    ├─ Calculate grid center (nose)
    ├─ Rotate grid (yaw/pitch)
    ├─ Build voxel centers (160 positions)
    └─ Calculate layer z-positions
    ↓
Update Hit Tracking (per hand)
    ├─ Calculate trigger point (palm center)
    ├─ Calculate relative_z (depth scaling)
    ├─ Select appropriate layer
    ├─ Find nearest voxel
    ├─ Record hit if within radius
    ├─ Update chain code (direction)
    ├─ Track palm angles
    └─ Track distance to nose
    ↓
Draw Grid
    ├─ Project voxels to 2D
    ├─ Color based on hits
    ├─ Render with depth ordering
    └─ Overlay indices and path
    ↓
Store Sample
    ├─ Save normalized points
    ├─ Save hit order
    ├─ Save chain code
    ├─ Save angles
    └─ Save distances
    ↓
Export Dataset
    ├─ JSON format (complete data)
    └─ CSV format (flattened)
```

---

## Key Design Properties

### Adaptive to Individual Differences
- Shoulder distance adapts to person size
- Face distance adapts to head position
- Grid centers on nose, not fixed position

### Invariant to Hand Speed
- Chain code records direction, not speed
- Hit triggering based on proximity, not velocity
- Works with slow and fast gestures equally

### Robust to Partial Visibility
- Continues tracking even if hand partially leaves frame
- Last known posture used if needed
- Multiple voxel triggers per frame

### Real-Time Performance
- Face Mesh: ~30 FPS
- Hand detection: ~30 FPS
- Grid calculations: <1ms per frame
- Suitable for live interaction

---

## Coordinate Transformations

### Raw Coordinates → Normalized

**Input:** MediaPipe outputs in pixel and relative depth coordinates

**Process:**
1. Convert pixel coordinates to [0, 1]
2. Normalize relative to face size
3. Apply head rotation corrections

**Output:** Normalized coordinates in [0, 1]

### Example
```
Raw: x=640 pixels, frame_width=1280 → Normalized: x=0.5
Raw: y=360 pixels, frame_height=720 → Normalized: y=0.5
Raw: z=-0.1 (MediaPipe) → Normalized: z=0.45 (relative to face)
```

---

## Module Interactions

### main.py ↔ capture.py
- Starts camera capture with `HandCapture.initialize()`
- Retrieves landmarks with `get_landmarks()`
- Manages capture state with `start/stop_grid_tracking()`

### capture.py ↔ face_grid_3d.py
- Processes frames with `FaceGrid3D.process_frame()`
- Updates hits with `update_hit_tracking()`
- Retrieves hits with `get_hit_order()`, `get_chain_code()`, etc.

### main.py ↔ dataset_io.py
- Adds samples with `add_sample()`
- Exports with `save_as_json()`, `save_as_csv()`
- Manages in-memory dataset

### main.py ↔ ui.py
- Displays preview with `update_preview_image()`
- Handles user input callbacks
- Updates sample count and table

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Face detection | ~30ms | MediaPipe Face Mesh |
| Hand detection | ~30ms | MediaPipe Hands (2 hands) |
| Voxel calculation | <1ms | 160 voxels |
| Hit detection | <1ms | 2D nearest-neighbor search |
| Grid rendering | ~5ms | OpenCV drawing |
| Full frame | ~70-100ms | 10-15 FPS with visualization |

---

## Extensions and Future Improvements

### Possible Extensions

1. **More Layers** - Expand from 2 to 3+ depth layers
2. **Variable Grid Size** - Support different resolutions
3. **Larger Hand Support** - Handle dual-hand interactions
4. **Gesture Recognition** - Real-time classification from hits
5. **Data Augmentation** - Synthetic gesture generation

### Optimization Opportunities

1. **SIMD Vectorization** - Faster voxel calculations
2. **Caching** - Cache common transformations
3. **Multi-threading** - Process pose separately from hands
4. **GPU Acceleration** - Use OpenCV GPU functions

---

## Design Rationale

### Why 2 Layers?
- Captures hand approaching and receding motion
- Clear distinction between near and far space
- Computational simplicity vs. expressiveness

### Why Face-Centered?
- Natural coordinate system for gestures
- Invariant to body position
- Simplifies data collection

### Why Adaptive Depth?
- Different people have different reaches
- One threshold can't work universally
- Dynamic scaling is automatic and robust

### Why 26-Direction Chain Code?
- Captures trajectory information
- Complementary to spatial hits
- Reduced data dimensionality
- Good for gesture recognition

---

## See Also

- [CORE_CONCEPTS.md](CORE_CONCEPTS.md) - Algorithm details and mathematical formulas
- [DEVELOPMENT.md](DEVELOPMENT.md) - Code implementation and APIs
- [USER_GUIDE.md](USER_GUIDE.md) - How to use the application

---

**Version:** January 2026  
**Status:** Active
