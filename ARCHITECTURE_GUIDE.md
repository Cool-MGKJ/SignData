# SignData Architecture Guide

**Purpose**: This guide must be referenced when updating ANY code in the SignData project. It ensures existing logic is preserved and future code builds on the previous logic.

---

## 1. Core Layer System

### Layer Hierarchy (Face-First Naming)
```
Face (MediaPipe landmarks)
  ↓
Layer 0 (z_idx=0, voxels 0–79, GREEN)
  Inner layer closest to face
  Triggered when: relative_z >= 0 (hand at/approaching face)
  ↓
Layer 1 (z_idx=1, voxels 80–159, RED)
  Outer layer closest to camera
  Triggered when: relative_z < 0 (hand extended toward camera)
  ↓
Camera (MediaPipe viewpoint)
```

### Critical Invariant: z_idx ↔ Layer Mapping
- **MUST ALWAYS HOLD**: `z_idx == layer_number`
- Layer 0 ALWAYS uses `z_idx=0` (voxels 0–79)
- Layer 1 ALWAYS uses `z_idx=1` (voxels 80–159)
- If you see `if z_idx == 0:`, it refers to Layer 0 (face)
- If you see `if z_idx == 1:`, it refers to Layer 1 (camera)

**Impact**: This naming convention prevents indexing bugs and makes code self-documenting. Any update that breaks this mapping introduces critical errors.

---

## 2. Depth Layer Spacing & Calculation

### How Layer Depth is Set

#### Parameters (in `FaceGrid3D.__init__`)
```python
depth_layers: int = 2              # Number of depth layers (2 = face + camera)
depth_span_factor: float = 2.4     # Multiplier controlling layer separation
```

#### Depth Calculation (in `process_frame()`)
```python
# 1. Calculate horizontal spacing (normalized units)
dx_norm = distance_from_nose_to_left_eye
# Typical value: 0.05–0.12 depending on face size

# 2. Calculate vertical spacing
dy_norm = eye_distance * 1.2
# Typical value: 0.10–0.25

# 3. Calculate depth layer thickness (Δz between layers)
layer_divisor = max(depth_layers - 1, 1)  # For 2 layers: divisor = 1
dz_norm = dx_norm * depth_span_factor / layer_divisor
#        = dx_norm * 2.4 / 1
#        = dx_norm * 2.4
```

#### Voxel Position Formula
For each voxel at grid indices (x_idx, y_idx, z_idx):
```python
offset_z = z_idx * dz_norm

# Layer 0 (z_idx=0): offset_z = 0 * dz_norm = 0        (at face depth)
# Layer 1 (z_idx=1): offset_z = 1 * dz_norm = dz_norm   (forward from face)

voxel_z = nose_depth + offset_z  # Relative to nose depth
```

### What Controls Layer Depth?

| Parameter | Purpose | Impact | Current Value |
|-----------|---------|--------|---------------|
| `depth_span_factor` | Multiplier for layer separation | Larger = farther apart layers | 2.4 |
| `dx_norm` | Horizontal spacing (derived from face size) | Larger face = larger separation | Auto-calculated |
| `depth_layers` | Number of layers | 2 = thin grid, 3+ = thicker grid | 2 |

### Example Calculation
If `dx_norm = 0.1` (10% of frame width) and `depth_span_factor = 2.4`:
```
dz_norm = 0.1 * 2.4 / 1 = 0.24
Layer 0 is at nose_depth + 0.0
Layer 1 is at nose_depth + 0.24
```

---

## 3. Dynamic Depth Scaling & Layer Triggering

### Hand Position to Layer Mapping (Complete Algorithm)

#### Step 1: Get Hand Position
```python
lx_norm, ly_norm, lz_mp = trigger_point  # From MediaPipe Hand Landmarks
# lz_mp is raw MediaPipe z-depth (negative = closer to camera, positive = farther)
```

#### Step 2: Calculate Reference Values (from current frame)
```python
# Reference length: body size (once per frame)
reference_length = shoulder_to_shoulder_distance  # From MediaPipe Pose
# Typical range: 0.15 - 0.35 (normalized units)

# Face Z reference: face plane zero-point (once per frame)
face_z_reference = nose_tip_z_from_mediapipe  # From MediaPipe Face Mesh
# Typical range: -0.3 to +0.3 (MediaPipe z-depth)
```

#### Step 3: Transform to Relative Z
```python
relative_z = (lz_mp - face_z_reference) / reference_length

Interpretation:
  • relative_z < 0  : Hand is EXTENDED TOWARD camera
  • relative_z = 0  : Hand is AT face plane
  • relative_z > 0  : Hand is APPROACHING face or BEYOND face plane
```

#### Step 4: Determine Layer (THRESHOLD = 0)
```python
if relative_z < 0:
    matching_layer_idx = 1  # Layer 1 (OUTER, z_idx=1, voxels 80–159)
    # Color: RED when hit
    # Meaning: Hand extended toward camera (first contact)
else:  # relative_z >= 0
    matching_layer_idx = 0  # Layer 0 (INNER, z_idx=0, voxels 0–79)
    # Color: GREEN when hit
    # Meaning: Hand at or approaching face (second contact)
```

#### Step 5: Hit Detection (within selected layer only)
```python
# Get all voxels in matching_layer_idx
for voxel in voxels_in_layer:
    # Calculate 2D distance (x,y only; z already matched)
    dist_2d = sqrt((lx_norm - voxel_x)^2 + (ly_norm - voxel_y)^2)
    
    if dist_2d <= hit_radius_norm:  # typically 0.12
        → Mark voxel as hit
        → Add to voxel_hit_order
        → Update chain code
```

### Key Invariants (MUST PRESERVE)
1. **z_idx = layer_number**: `z_idx=0` is Layer 0, `z_idx=1` is Layer 1
2. **Threshold is 0**: Layer decision point is always at `relative_z = 0`
3. **Only 2D distance**: Hit detection uses only x,y; z is pre-matched by layer selection
4. **Single voxel per frame**: Only nearest voxel in selected layer is hit
5. **No fallback zones**: Never use hardcoded z-ranges or distance-based thresholds

### Reference Length Calculation Priority
```python
# Priority 1: Shoulder-to-shoulder (most stable)
if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
    reference_length = distance_3d(left_shoulder, right_shoulder)

# Priority 2: Wrist-to-elbow (fallback)
elif wrist.visibility > 0.5 and elbow.visibility > 0.5:
    reference_length = distance_3d(wrist, elbow)

# Else: No valid reference (skip hand)
```

### Visual Representation
```
                      CAMERA (MediaPipe viewpoint)
                              ↑
                              |
                         relative_z < 0
                    (LAYER 1 TRIGGERED - RED)
                              |
                    ┌─────────────────────┐
                    │   LAYER 1 (RED)     │
                    │   z_idx=1           │  ← Hand extended toward camera
                    │   voxels 80–159     │
                    └─────────────────────┘
                              |
                    ════════════════════  ← relative_z = 0 (THRESHOLD)
                    Threshold / Face Plane
                    ════════════════════
                              |
                    ┌─────────────────────┐
                    │   LAYER 0 (GREEN)   │  ← Hand at/approaching face
                    │   z_idx=0           │
                    │   voxels 0–79       │
                    └─────────────────────┘
                              |
                         relative_z >= 0
                    (LAYER 0 TRIGGERED - GREEN)
                              |
                       YOUR FACE (Nose)
```

### Reference Length & Face Z Reference Calculation
- **Reference Length** (distance-invariance):
  - Primary: `shoulder_distance = distance_3d(left_shoulder, right_shoulder)` (MediaPipe Pose)
  - Fallback: `arm_length = distance_3d(wrist, elbow)` (MediaPipe Pose)
  - Purpose: Scales relative_z calculation to be independent of how close/far you are from camera
  - Typical range: 0.15–0.35 (normalized units)

- **Face Z Reference** (zero-point calibration):
  - Value: `nose_z = mediapipe_face_mesh.landmark[4].z` (Nose Tip)
  - Purpose: Sets the zero-point (face plane) for relative_z calculation
  - Updates every frame to track head movement
  - Typical range: -0.3 to +0.3 (MediaPipe z-depth)

### Gesture Flow Example
```
User moves hand from camera toward face:

1. Hand far away (relative_z = -0.5):
   → Layer 1 (RED) voxels trigger
   → voxel_hit_order: [95, 96, 107, ...]

2. Hand moving closer (relative_z = -0.2):
   → Still Layer 1 (RED)
   → voxel_hit_order: [..., 108, 119, ...]

3. Hand crosses face plane (relative_z ≈ 0):
   → Transition from Layer 1 to Layer 0
   → Color changes from RED to GREEN

4. Hand near/at face (relative_z = +0.1):
   → Layer 0 (GREEN) voxels trigger
   → voxel_hit_order: [..., 45, 34, 23, ...]

Final hit order captures entire path from camera → face
```

---

## 4. Voxel Hit Detection

### Hit Radius Parameter
```python
hit_radius_norm: float = 0.12  # Detection radius in normalized 3D space
```

### Detection Algorithm
```
For each hand trigger point (lx_norm, ly_norm, lz_mp):
  1. Calculate relative_z = (lz_mp - face_z_ref) / ref_length
  2. Determine matching_layer_idx (see Section 3)
  3. Get all voxels in matching_layer_idx only
  4. For each voxel in that layer:
     - Calculate 2D distance: dist_2d = sqrt((lx - vx)² + (ly - vy)²)
     - If dist_2d <= hit_radius_norm:
       → Mark voxel as hit
       → Append to voxel_hit_order
       → Update chain code (if capturing)
```

**Important**: 
- Hit detection uses **2D distance only** (x, y), not 3D
- Z-coordinate already matched via layer selection
- Only one voxel per frame per hand is hit (nearest voxel)

---

## 5. Chain Code System

### Direction Vectors (26-Directional)
```python
# 6 face directions (orthogonal)
±x, ±y, ±z

# 12 edge directions (diagonal on one plane)
±x±y, ±x±z, ±y±z

# 8 corner directions (all 3 axes)
±x±y±z
```

### Chain Code Generation
```
When a new voxel is hit:
  1. Get previous voxel position (from voxel_hit_order)
  2. Get current voxel position
  3. Calculate direction vector = current - previous
  4. Normalize the direction vector to unit length
  5. Find closest match among 26 directions (using cosine similarity)
  6. Append direction_index (0-25) to current_gesture_chain
```

**Storage**: Chain code is a list of integers [0-25] representing movement trajectory.

---

## 6. Visualization Color Scheme

### Layer Colors (Hit State)
| Layer | z_idx | Color (Hit) | Color (Unhit) | Comments |
|-------|-------|-----------|---------------|----------|
| Layer 0 | 0 | Bright Green `(0,255,0)` | Dim Green | Face-closest layer |
| Layer 1 | 1 | Bright Red `(0,0,255)` | Dim Red | Camera-closest layer |

### Drawing Order
- Voxels sorted by z-depth (farthest first, nearer on top)
- Each layer offset horizontally by 3 pixels for visual separation

---

## 7. Coordinate Systems

### 1. MediaPipe Normalized (Input)
```
Range: 0.0 to 1.0
Origin: Top-left of frame
Axes:
  x: 0 (left) → 1 (right)
  y: 0 (top) → 1 (bottom)
  z: relative depth (-0.5 to +0.5), where:
     negative z = closer to camera
     positive z = farther from camera
```

### 2. Grid Normalized (Internal)
```
Range: 0.0 to 1.0
Origin: Face center (nose)
Axes:
  x: left ← 0 → right
  y: top ← 0 → bottom
  z: relative to face depth
     Layer 0: z ≈ 0
     Layer 1: z ≈ dz_norm (typically 0.24)
```

### 3. Pixel Coordinates (Output/Visualization)
```
Range: 0 to frame_width/height
Origin: Top-left of frame
Conversion: (u, v) = (x_norm * frame_width, y_norm * frame_height)
```

---

## 8. Key Data Structures

### `voxel_centers` (List of Tuples)
```python
[
  (x_norm_0, y_norm_0, z_norm_0),  # Voxel 0
  (x_norm_1, y_norm_1, z_norm_1),  # Voxel 1
  ...
  (x_norm_159, y_norm_159, z_norm_159)  # Voxel 159
]
```
- Order: [z, row, col] (z fastest, then row, then col)
- Layer 0: indices 0–79
- Layer 1: indices 80–159

### `voxel_hit_bool` (Boolean Array)
```python
[True, False, True, ..., False]  # Length 160
# True = voxel was hit during capture
```

### `voxel_hit_order` (List of Integers)
```python
[15, 28, 45, 15, 52, ...]
# Ordered sequence of voxel indices in the order they were hit
```

### `current_gesture_chain` (List of Integers)
```python
[0, 1, 4, 6, 18, ...]
# Directional trajectory (0-25 direction indices)
```

---

## 9. Maintenance Guidelines

### When Updating Trigger Logic
- **PRESERVE**: The `relative_z` calculation and interpretation
- **PRESERVE**: The z_idx ↔ Layer mapping
- **PRESERVE**: Layer color assignments (Layer 0 = green, Layer 1 = red)
- **TEST**: Run with hand moving from camera toward face, verify Layer 1 triggers first, then Layer 0

### When Updating Visualization
- **PRESERVE**: Layer-to-color mapping
- **PRESERVE**: Depth-based drawing order (back-to-front)
- **MODIFY**: Only radius, transparency, or overlay styling

### When Updating Depth Calculation
- **DOCUMENT**: Any changes to `depth_span_factor` or layer spacing logic
- **VALIDATE**: Ensure both layers remain visible in preview
- **TEST**: Verify hit detection still works across both layers

### When Adding New Features
1. Identify which layer system component it touches (triggering, visualization, storage, etc.)
2. Check this guide for that component's invariants
3. Ensure new code preserves ALL invariants
4. Add inline comments explaining non-obvious logic
5. Update this guide if adding new components

---

## 10. Testing Checklist

### Layer Triggering Test
- [ ] Hand far from face: Layer 1 (red) voxels trigger
- [ ] Hand approaching face: Layer 0 (green) voxels trigger
- [ ] No voxels trigger outside active layer
- [ ] Hit order appended in sequence

### Visualization Test
- [ ] Layer 0 voxels appear green (unhit) and bright green (hit)
- [ ] Layer 1 voxels appear red (unhit) and bright red (hit)
- [ ] Path line connects hit voxels in order
- [ ] Trigger point indicator visible (red circle with white outline)

### Depth Spacing Test
- [ ] Two distinct layers visible at different depths
- [ ] Layers maintain spacing as head moves
- [ ] No voxel overlap between layers

### Chain Code Test
- [ ] Chain code generates for hand movement
- [ ] Direction indices in range 0–25
- [ ] Same direction not repeated consecutively
- [ ] Chain code exported with dataset

---

## 11. Quick Reference: Most Common Updates

### Change Layer Triggering Behavior
**File**: `face_grid_3d.py`, method `update_hit_tracking()` (lines ~937–945)
**Invariant**: Preserve z_idx ↔ Layer mapping and relative_z logic

### Change Layer Colors
**File**: `face_grid_3d.py`, method `draw_grid()` (lines ~1028–1034)
**Invariant**: Layer 0 must be distinguishable from Layer 1

### Change Layer Depth Separation
**File**: `face_grid_3d.py`, parameter `depth_span_factor` (line ~123)
**Invariant**: Update `dz_norm = dx_norm * depth_span_factor / layer_divisor`

### Change Hit Radius
**File**: `face_grid_3d.py`, parameter `hit_radius_norm` (line ~123)
**Invariant**: Should be proportional to grid spacing

---

## 12. Version History

| Date | Change | By | Notes |
|------|--------|-----|-------|
| 2026-01-13 | Initial Architecture Guide | — | Established face-first layer naming, depth calculation docs |
| | | | |

---

**Last Updated**: 2026-01-13  
**Reviewed By**: —  
**Status**: Active — Reference for all future updates
