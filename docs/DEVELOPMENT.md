# Development Guide: Code Structure and APIs

Complete reference for developers working with the SignData codebase.

## Module Overview

```
SignData/
├── main.py              Main application coordinator
├── capture.py           Camera and hand detection
├── face_grid_3d.py      3D voxel grid tracker
├── dataset_io.py        Data storage and export
├── ui.py                User interface
├── normalization.py     Hand landmark normalization
└── docs/                Documentation (this folder)
```

---

## Module: main.py

**Purpose:** Orchestrates capture, processing, saving, and export

### Class: ASLDataCollectionApp

Main application class coordinating all components.

#### Key Methods

| Method | Purpose |
|--------|---------|
| `start_capture()` | Begin gesture recording |
| `stop_capture()` | End gesture recording and process |
| `save_sample(label)` | Save current sample with label |
| `export_dataset(filepath, format)` | Export to JSON/CSV |
| `clear_session()` | Clear all samples |
| `update_video()` | Main loop - process frames |

#### State Variables

```python
self.is_capturing : bool                      # Capture mode active
self.captured_landmarks : List[dict]          # Current hand landmarks
self.captured_hit_order : List[int]           # Voxel hit sequence
self.captured_chain_code : List[int]          # Directional trajectory
self.captured_palm_angles_left : List[tuple]  # Left hand angles
self.captured_palm_angles_right : List[tuple] # Right hand angles
self.captured_trigger_distance_left : List    # Left hand distance
self.captured_trigger_distance_right : List   # Right hand distance
```

#### Example Usage

```python
app = ASLDataCollectionApp()

# Start capturing
app.start_capture()

# ... frames processed ...

# Stop capturing
app.stop_capture()

# Save with label
app.save_sample("hello")

# Export
app.export_dataset("dataset.json", "json")
```

---

## Module: capture.py

**Purpose:** Manages camera input and MediaPipe hand/pose detection

### Class: HandCapture

Handles webcam capture and landmark detection.

#### Key Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `initialize()` | bool | Initialize camera (must call first) |
| `process_frame()` | bool | Capture and process one frame |
| `get_landmarks(frame)` | List[dict] | Extract hand landmarks from frame |
| `get_hit_order()` | List[int] | Get voxel hit sequence |
| `get_chain_code()` | List[int] | Get directional trajectory |
| `get_palm_angles_left()` | List[tuple] | Get left hand angle changes |
| `get_palm_angles_right()` | List[tuple] | Get right hand angle changes |
| `get_trigger_distance_left()` | List[float] | Get left hand distance changes |
| `get_trigger_distance_right()` | List[float] | Get right hand distance changes |
| `start_grid_tracking()` | None | Begin recording voxel hits |
| `stop_grid_tracking()` | None | End recording voxel hits |
| `release()` | None | Clean up resources |

#### Landmark Format

Returned by `get_landmarks(frame)`:
```python
[
    {
        'handedness': 'Left' or 'Right',
        'landmarks': [
            (x0, y0, z0),  # wrist
            (x1, y1, z1),  # thumb_cmc
            ...
            (x20, y20, z20) # pinky_tip
        ]
    },
    ...
]
```

Where x, y, z are in normalized coordinates [0, 1].

#### Example Usage

```python
capture = HandCapture(camera_index=0)
if not capture.initialize():
    print("Camera initialization failed")
    exit()

capture.start_grid_tracking()

while True:
    if not capture.process_frame():
        break
    
    landmarks = capture.get_landmarks(capture.current_frame)
    hit_order = capture.get_hit_order()
    chain_code = capture.get_chain_code()

capture.stop_grid_tracking()
capture.release()
```

---

## Module: face_grid_3d.py

**Purpose:** Core 3D voxel grid tracking and visualization

### Class: FaceGrid3D

Manages face-centered 3D voxel grid.

#### Constructor Parameters

```python
FaceGrid3D(
    breadth: int = 8,           # X-axis voxels (width)
    length: int = 10,           # Y-axis voxels (height)
    depth_layers: int = 2,      # Z-axis layers
    depth_span_factor: float = 2.4,      # Depth scale
    hit_radius_norm: float = 0.12,       # Hit detection radius
    track_landmark_paths: bool = True,   # Record paths
    tracked_landmarks: List[int] = None  # Which landmarks to track
)
```

#### Grid Dimensions

- **Total Voxels:** breadth × length × depth_layers (default: 160)
- **Voxel Indexing:** `idx = z * (breadth * length) + y * breadth + x`
- **Layer 0:** Voxels 0-79 (inner, near face)
- **Layer 1:** Voxels 80-159 (outer, near camera)

#### Key Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `process_frame(frame)` | bool | Build grid from face landmarks |
| `update_hit_tracking(hand_landmarks)` | None | Record hand hits |
| `draw_grid(frame, show_hits)` | ndarray | Render grid visualization |
| `get_hit_order()` | List[int] | Get voxel hit sequence |
| `get_hit_grid_3d()` | List[int] | Get binary hit vector [0,1] |
| `get_chain_code()` | List[int] | Get directional codes |
| `get_palm_angles_left()` | List[tuple] | Get left hand (yaw, pitch, roll) |
| `get_palm_angles_right()` | List[tuple] | Get right hand (yaw, pitch, roll) |
| `get_trigger_distance_left()` | List[float] | Get left hand-to-nose distances |
| `get_trigger_distance_right()` | List[float] | Get right hand-to-nose distances |
| `get_nose_position()` | Optional[tuple] | Get nose (x, y, z) |
| `reset_hit_tracking()` | None | Clear all tracking data |

#### State Variables (Read-Only)

```python
self.voxel_centers : List[tuple]        # 3D positions of voxels
self.voxel_centers_2d : List[tuple]     # 2D pixel positions
self.voxel_hit_bool : List[bool]        # Which voxels were hit
self.voxel_hit_order : List[int]        # Order of hits
self.current_gesture_chain : List[int]  # Chain code
self.head_yaw_deg : float               # Head yaw angle
self.head_pitch_deg : float             # Head pitch angle
```

#### Example Usage

```python
grid = FaceGrid3D(breadth=8, length=10, depth_layers=2)

# Process frame with face landmarks
grid.process_frame(frame)

# Track hand movements
grid.update_hit_tracking(hand_landmarks_list)

# Get results
hit_order = grid.get_hit_order()
chain_code = grid.get_chain_code()
angles_left = grid.get_palm_angles_left()
distances_right = grid.get_trigger_distance_right()

# Visualize
annotated_frame = grid.draw_grid(frame, show_hits=True)
```

#### Voxel Coordinate Conversion

```python
# Get voxel position
x_idx, y_idx, z_idx = 4, 5, 0  # (x=4, y=5, layer=0)
linear_idx = grid.voxel_index(x_idx, y_idx, z_idx)
x, y, z = grid.voxel_centers[linear_idx]  # Normalized position

# Pixel position
pixel_x, pixel_y = grid.voxel_centers_2d[linear_idx]
```

---

## Module: dataset_io.py

**Purpose:** In-memory dataset storage and file export

### Class: DatasetManager

Manages collected samples and file I/O.

#### Key Methods

| Method | Purpose |
|--------|---------|
| `add_sample(...)` | Add new sample to dataset |
| `get_all_samples()` | Return list of all samples |
| `get_sample_count()` | Return number of samples |
| `clear()` | Remove all samples |
| `save_as_json(path)` | Export to JSON file |
| `save_as_csv(path)` | Export to CSV file |

#### add_sample() Signature

```python
def add_sample(
    label: str,                                  # Sign name
    normalized_points: List[Tuple[float, float, float]],  # Hand coords
    hand: str = "unknown",                       # "left", "right", "both"
    hit_order: Optional[List[int]] = None,       # Voxel sequence
    chain_code: Optional[List[int]] = None,      # Trajectory
    palm_angles_left: Optional[List[tuple]] = None,      # Yaw/pitch/roll
    palm_angles_right: Optional[List[tuple]] = None,     # Yaw/pitch/roll
    trigger_distance_left: Optional[List[float]] = None, # Distances
    trigger_distance_right: Optional[List[float]] = None # Distances
) -> int:
    """Returns: Sample ID assigned to this sample"""
```

#### Sample Data Structure

```python
sample = {
    'id': 1,
    'label': 'hello',
    'hand': 'right',
    'points': [
        {'index': 0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
        ...
    ],
    'num_points': 21,
    'timestamp': '2026-01-14T13:35:09.881392',
    'hit_order': [31, 23, 103, ...],
    'chain_code': [3, 4, 15, ...],
    'palm_angles_left': [[10.2, 5.3, -2.1], ...],
    'palm_angles_right': [],
    'trigger_distance_left': [],
    'trigger_distance_right': [0.45, 0.48, ...]
}
```

#### Example Usage

```python
dataset = DatasetManager()

# Add samples
sample_id = dataset.add_sample(
    label='hello',
    normalized_points=[(0, 0, 0), (-0.34, 0.24, -0.16), ...],
    hand='right',
    hit_order=[31, 23, 103, ...],
    chain_code=[3, 4, 15, ...],
    palm_angles_left=[],
    palm_angles_right=[[10.2, 5.3, -2.1], [11.5, 6.1, -1.8]],
    trigger_distance_left=[],
    trigger_distance_right=[0.45, 0.48, ...]
)

# Export
dataset.save_as_json('data.json')
dataset.save_as_csv('data.csv')

# Check
print(f"Total samples: {dataset.get_sample_count()}")
```

---

## Module: ui.py

**Purpose:** Tkinter-based user interface

### Class: ASLDataCollectionUI

Manages the desktop UI.

#### Key Methods

| Method | Purpose |
|--------|---------|
| `set_callbacks(...)` | Register button callbacks |
| `update_preview_image(frame)` | Update camera preview |
| `add_sample_to_table(...)` | Add row to samples table |
| `update_sample_count(count)` | Update count display |
| `get_label_input()` | Get user-entered label |

#### Callbacks

```python
ui.set_callbacks(
    on_start_capture=start_capture_func,
    on_stop_capture=stop_capture_func,
    on_save_sample=save_sample_func,
    on_export_dataset=export_dataset_func,
    on_clear_session=clear_session_func
)
```

---

## Module: normalization.py

**Purpose:** Hand landmark normalization and preprocessing

### Key Functions

#### normalize_hand_data()

```python
def normalize_hand_data(
    hand_landmarks: List[Tuple[float, float, float]],
    apply_rotation: bool = True
) -> List[Tuple[float, float, float]]:
    """
    Normalize hand landmarks to consistent coordinate system.
    
    Steps:
    1. Center on wrist (landmark 0)
    2. Scale by average finger distance
    3. Optionally rotate to standard orientation
    
    Returns: Normalized landmarks in [-1, 1] range
    """
```

#### get_hand_info()

```python
def get_hand_info(hand_landmarks_list: List[dict]) -> str:
    """
    Determine which hand(s) are present.
    
    Returns: "left", "right", "both", or "none"
    """
```

---

## Common Workflows

### Workflow 1: Capture One Gesture

```python
# Initialize
app = ASLDataCollectionApp()

# Start capture
app.start_capture()

# ... user performs gesture ...

# Stop and save
app.stop_capture()
app.save_sample("hello")
```

### Workflow 2: Batch Capture

```python
labels = ["hello", "goodbye", "thank_you"]

for label in labels:
    app.start_capture()
    # ... user performs gesture ...
    app.stop_capture()
    app.save_sample(label)
    # ... repeat 5 times for each sign ...

app.export_dataset("batch1.json", "json")
```

### Workflow 3: Process Saved Data

```python
# Load from file
with open("data.json") as f:
    data = json.load(f)

# Analyze
for sample in data['samples']:
    label = sample['label']
    hit_count = len(sample['hit_order'])
    chain_len = len(sample['chain_code'])
    angles_left = len(sample['palm_angles_left'])
    distances = len(sample['trigger_distance_right'])
    
    print(f"{label}: {hit_count} hits, "
          f"{chain_len} directions, "
          f"{angles_left} angle changes")
```

---

## Key Constants and Configuration

### Grid Parameters

```python
BREADTH = 8              # X-axis voxels
LENGTH = 10             # Y-axis voxels
DEPTH_LAYERS = 2        # Z-axis layers
NUM_VOXELS = 160        # Total voxels
HIT_RADIUS = 0.12       # Hit detection radius (normalized)
```

### Threshold Parameters

```python
ANGLE_CHANGE_THRESHOLD = 5.0      # degrees
DISTANCE_CHANGE_THRESHOLD = 0.05  # normalized units
```

### Coordinate Ranges

```python
X_RANGE = [0, 1]        # Normalized X
Y_RANGE = [0, 1]        # Normalized Y
Z_RANGE = [0, 1]        # Normalized depth
```

### Head Rotation Limits

```python
MAX_YAW = 40.0           # degrees
MAX_PITCH = 30.0         # degrees
```

---

## Error Handling

### Common Exceptions

```python
try:
    capture.initialize()
except Exception as e:
    print(f"Camera initialization failed: {e}")
    # Handle gracefully

try:
    landmarks = capture.get_landmarks(frame)
except AttributeError:
    # No hands detected
    landmarks = []

try:
    dataset.save_as_json(filepath)
except IOError as e:
    print(f"Save failed: {e}")
    # Handle file write error
```

### Debugging

Enable console output:
```python
# face_grid_3d.py prints layer info
print(f"[Layer {idx}] Relative_Z={relative_z:.3f}")

# dataset_io.py prints sample info
print(f"Added sample: ID={id}, Points={count}")

# main.py prints state transitions
print(f"Capture started/stopped")
```

---

## Performance Tips

### Optimization Opportunities

1. **Cache voxel centers** - Pre-compute, don't recalculate
2. **Use numpy operations** - Vectorized > loops
3. **Reduce drawing calls** - Batch OpenCV operations
4. **Skip frames** - Process every Nth frame if needed
5. **Lazy loading** - Only compute angles/distances when needed

### Profiling

```python
import time

start = time.time()
grid.process_frame(frame)
elapsed = time.time() - start
print(f"Frame processing: {elapsed*1000:.1f}ms")
```

---

## Testing

### Unit Test Template

```python
def test_voxel_indexing():
    grid = FaceGrid3D()
    idx = grid.voxel_index(4, 5, 0)
    assert idx == 4 + 5*8  # y*breadth + x

def test_normalization():
    landmarks = [(0, 0, 0)] * 21
    normalized = normalize_hand_data(landmarks)
    assert len(normalized) == 21
```

### Integration Test Template

```python
def test_capture_and_save():
    capture = HandCapture()
    dataset = DatasetManager()
    
    capture.start_grid_tracking()
    # ... simulate frames ...
    capture.stop_grid_tracking()
    
    sample_id = dataset.add_sample(
        label='test',
        normalized_points=[(0,0,0)]*21,
        hit_order=[1, 2, 3],
        chain_code=[0, 1, 2]
    )
    
    assert dataset.get_sample_count() == 1
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design overview
- [CORE_CONCEPTS.md](CORE_CONCEPTS.md) - Algorithms and formulas
- [USER_GUIDE.md](USER_GUIDE.md) - How to use the application

---

**Version:** January 2026  
**Status:** Active
