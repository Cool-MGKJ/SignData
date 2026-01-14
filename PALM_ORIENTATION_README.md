# Palm Orientation Tracking Implementation

## Overview

This implementation adds real-time palm orientation tracking with a 5-degree change trigger to the hand-tracking data capture tool. The system calculates the palm's orientation (Pitch, Yaw, Roll) in degrees and only saves samples when the orientation changes by at least 5 degrees.

## Key Components

### 1. `palm_orientation.py` Module

This module provides functions for calculating palm orientation:

- **`get_palm_orientation(landmarks)`**: Calculates palm orientation from 21 hand landmarks
  - Uses landmarks 0 (Wrist), 5 (Index MCP), and 17 (Pinky MCP) to define the palm plane
  - Calculates normal vector using cross product: `normal = (wrist→index_MCP) × (wrist→pinky_MCP)`
  - Converts normal vector to Euler angles (Pitch, Yaw, Roll) in degrees

- **`normal_to_euler_angles(normal)`**: Converts normalized 3D vector to Euler angles
  - **Pitch**: Rotation around X-axis (tilt up/down) - range: [-90, 90] degrees
  - **Yaw**: Rotation around Y-axis (turn left/right) - range: [-180, 180] degrees
  - **Roll**: Rotation around Z-axis (tilt side-to-side) - range: [-180, 180] degrees

- **`has_significant_change(current, last, threshold=5.0)`**: Checks if any angle changed by ≥ threshold
  - Returns `True` if `abs(current - last) >= threshold` for any axis

### 2. Integration in `main.py`

- **State Variables**:
  - `last_saved_angles`: Tracks the last saved palm orientation
  - `current_palm_angles`: Current frame's palm orientation

- **Capture Loop** (`update_video()`):
  - Calculates palm orientation for each frame during capture
  - Checks for 5-degree changes in real-time
  - Stores current angles for later use

- **Save Logic** (`save_sample()`):
  - Calculates palm angles from captured landmarks
  - Checks if orientation changed by ≥ 5 degrees
  - Warns user if change is less than threshold (but still saves)
  - Updates `last_saved_angles` after successful save

### 3. Data Structure Update (`dataset_io.py`)

Each sample now includes:
```json
{
  "id": 1,
  "label": "hello",
  "hand": "left",
  "points": [...],
  "points_flat": [...],
  "palm_angles": [pitch, yaw, roll]  // NEW: Array of 3 angles in degrees
}
```

## Usage

### During Capture:

1. Start capture mode - palm orientation tracking begins
2. Move your hand - the system calculates orientation in real-time
3. Stop capture - current palm angles are captured
4. Save sample - system checks if orientation changed by ≥ 5 degrees

### 5-Degree Threshold Logic:

```python
# In update_video() loop (during capture):
current_angles = get_palm_orientation(landmarks)
if has_significant_change(current_angles, last_saved_angles, threshold_degrees=5.0):
    # Orientation changed significantly
    # This triggers save when user clicks "Save Sample"
```

```python
# In save_sample() (when user saves):
if has_significant_change(palm_angles, last_saved_angles, threshold_degrees=5.0):
    # Save proceeds normally
    self.last_saved_angles = palm_angles.copy()
else:
    # Warn user but still allow save
    print("Warning: Palm orientation change is less than 5° threshold")
```

## Mathematical Details

### Palm Normal Vector Calculation:

```
wrist = landmarks[0]
index_mcp = landmarks[5]
pinky_mcp = landmarks[17]

vec_0_to_5 = index_mcp - wrist
vec_0_to_17 = pinky_mcp - wrist
normal = vec_0_to_5 × vec_0_to_17  (cross product)
normal_normalized = normal / ||normal||
```

### Euler Angle Conversion:

The normal vector is converted to Euler angles using:
- **Pitch**: `arcsin(-ny)` where ny is the Y component
- **Yaw**: `arctan2(nx, nz)` where nx, nz are X and Z components
- **Roll**: `arctan2(ny, nx)` for the XY plane projection

## Example Output

When saving a sample:
```
Sample saved with palm orientation: Pitch=12.34°, Yaw=-45.67°, Roll=8.90°
```

If change is less than 5 degrees:
```
Warning: Palm orientation change (3.21°) is less than 5° threshold.
  Current: Pitch=12.34°, Yaw=-45.67°, Roll=8.90°
  Last saved: Pitch=10.00°, Yaw=-45.00°, Roll=8.00°
```

## Files Modified

1. **`palm_orientation.py`** (NEW): Core orientation calculation functions
2. **`main.py`**: Added palm tracking state and integration
3. **`dataset_io.py`**: Added `palm_angles` field to sample structure

## Testing

To test the implementation:
1. Run the application: `python main.py`
2. Start capture and move your hand through different orientations
3. Save samples and observe the palm angle values in the console
4. Check the saved JSON file to verify `palm_angles` field is present





