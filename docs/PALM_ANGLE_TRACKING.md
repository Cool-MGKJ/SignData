# Palm Angle Tracking Feature

## Overview
Real-time palm orientation angle calculation and change detection system for the FaceGrid3D voxel tracking system.

**Status**: ✓ Complete and integrated  
**Threshold**: 5 degrees for recording significant changes  
**Noise Floor**: 0.1 degrees

---

## Quick Start

### Access Palm Angles
```python
# During or after gesture capture:
grid = FaceGrid3D(...)
grid.update_hit_tracking(hand_landmarks_list)

# Retrieve recorded angles
left_angles = grid.get_palm_angles("left")      # List of all recorded angles
left_changes = grid.get_palm_angle_change_history("left")  # (angle, delta) tuples

# or get both hands
all_angles = grid.get_palm_angles()        # {"left": [...], "right": [...]}
all_changes = grid.get_palm_angle_change_history()
```

### Enable Debug Output
Uncomment lines in `update_hit_tracking()` method around line 1165:
```python
# if palm_angle_info and palm_angle_info['recorded']:
#     print(f"[{hand_type.upper()} Palm] Angle={palm_angle_info['angle_deg']:.1f}° "
#           f"Change={palm_angle_info['angle_change_deg']:.1f}° (Recorded)")
```

---

## How It Works

### Angle Calculation
1. Extract hand landmarks from MediaPipe (21 per hand)
2. Use wrist and MCP (metacarpophalangeal) joints to define palm plane
3. Calculate normal vector to palm plane via cross product
4. Compute angle relative to world vertical (Y-axis)
5. Return angle in degrees (0-180 range)

```
MediaPipe Hand Landmarks Used:
  - Index 0: Wrist
  - Index 5: Index MCP
  - Index 9: Middle MCP
  - Index 13: Ring MCP
  - Index 17: Pinky MCP

Palm Plane: Defined by vectors (wrist→middle) and (wrist→ring)
Palm Normal: cross(vec1, vec2)
Angle: arccos(dot(palm_normal, [0,1,0])) in degrees
```

### Change Detection
1. Frame N: Calculate palm angle
2. Compare with previous frame's angle
3. Calculate angle delta (with wraparound handling at 0°/360°)
4. **If delta > 5.0°**: Record to `palm_angle_change_history`
5. **If delta > 0.1° and ≤ 5.0°**: Update internal state (don't record)
6. **If delta ≤ 0.1°**: Ignore (noise floor)

### Wraparound Handling
Correctly handles transitions near 0°/360°:
- 359° → 1° = 2° change (not 358°)
- 180° → 181° = 1° change

---

## Data Structures

### Tracking Variables (in `FaceGrid3D.__init__()`)
```python
self.palm_angle_threshold = 5.0          # degrees - threshold for recording
self.palm_angles = {}                     # Dict[hand_type] → List[float]
self.last_palm_angle = {}                 # Dict[hand_type] → float
self.palm_angle_change_history = {}       # Dict[hand_type] → List[(float, float)]
self.angle_change_tolerance = 0.1         # degrees - noise filter
```

### Return Format from `update_palm_angle()`
```python
{
    "hand": "right",                           # "left" or "right"
    "angle_deg": 45.2,                        # Current palm angle
    "angle_change_deg": 7.5,                  # Change from previous
    "recorded": True,                         # Whether change was recorded
    "reason": "exceeded_threshold_5.0deg"     # or "below_threshold_5.0deg", "initial"
}
```

---

## Method Reference

### `calculate_palm_angle(hand_landmarks) → Optional[float]`
**Computes palm orientation angle from hand landmarks**

- **Input**: List of 21 (x, y, z) tuples from MediaPipe Hand
- **Output**: Angle in degrees (0-180), or None if invalid
- **Performance**: O(1)
- **Error Handling**: Returns None if:
  - Fewer than 21 landmarks
  - Landmarks too close to parallel (normal calculation invalid)

**Algorithm**:
```python
1. Extract: wrist, index_mcp, middle_mcp, ring_mcp landmarks
2. vec1 = middle_mcp - wrist
3. vec2 = ring_mcp - wrist
4. normal = cross(vec1, vec2)
5. normal_normalized = normal / ||normal||
6. angle = arccos(dot(normal_normalized, [0,1,0]))
7. return degrees(angle), clamped to 0-180
```

### `update_palm_angle(hand_type, hand_landmarks) → Optional[Dict]`
**Tracks angle changes and records significant changes**

- **Input**: 
  - `hand_type`: "left" or "right"
  - `hand_landmarks`: List of 21 landmarks
- **Output**: Dict with angle info, or None if calculation failed
- **Side Effects**: Updates internal tracking dictionaries
- **Threshold Logic**:
  - `delta > 5.0°`: Record to history
  - `0.1° < delta ≤ 5.0°`: Update state only
  - `delta ≤ 0.1°`: Ignore (noise)

### `get_palm_angles(hand_type=None) → Dict`
**Retrieve recorded palm angles**

- **Input**: "left", "right", or None (both)
- **Output**: Dict[hand_type] → List[float]

### `get_palm_angle_change_history(hand_type=None) → Dict`
**Retrieve angle change history**

- **Input**: "left", "right", or None (both)
- **Output**: Dict[hand_type] → List[(angle_deg, change_deg)]

---

## Integration in Update Pipeline

Called in `update_hit_tracking()` after hit tracking completes:

```python
for hand_data in hand_landmarks_list:
    # ... hit tracking and chain code ...
    
    # Track palm angle changes for this hand
    hand_type = hand_data.get('hand_type', 'unknown')
    if hand_type in ['left', 'right']:
        palm_angle_info = self.update_palm_angle(hand_type, landmarks)
        # Optional: print(f"[{hand_type.upper()} Palm] ...")
```

**Execution**: Once per hand per frame, after voxel hit tracking

---

## Use Cases

### 1. Gesture Classification
Distinguish hand poses:
- **Rotating hand**: Multiple recorded palm angle changes during gesture
- **Static hand**: No recorded angle changes
- **Mirrored gesture**: Identical angle change sequences for left/right

### 2. Orientation Tracking
Monitor hand orientation transitions:
- **Palm-up to palm-down**: Large angle change (e.g., 170°+ delta)
- **Subtle rotation**: Small angle changes within 5-10° range

### 3. Hand Morphology
Capture hand shape variations:
- Different hand shapes may have different angle baselines
- Pronation/supination clearly recorded

### 4. Motion Analysis
Extract temporal rotation patterns:
- Identify when hand starts rotating
- Measure rotation velocity (angle change per frame)
- Detect multi-part rotations vs. single-part

---

## Customization

### Change Threshold
```python
grid = FaceGrid3D(...)
grid.palm_angle_threshold = 10.0  # Record changes > 10 degrees instead
```

### Noise Tolerance
```python
grid.angle_change_tolerance = 0.05  # Stricter noise filtering
```

### Enable/Disable
```python
# To effectively disable: set threshold very high
grid.palm_angle_threshold = 999.0

# Or check before recording in your analysis
if angle_info['recorded']:
    # Process significant angle change
```

---

## Testing Recommendations

### 1. Angle Calculation Accuracy
Test with known hand orientations:
```
Palm parallel to camera (facing camera):  ~0-20°
Palm perpendicular to camera (side):      ~70-90°
Palm facing away (back of hand):          ~160-180°
```

### 2. Threshold Sensitivity
Start with 5° threshold; adjust if needed:
- **Too strict** (< 2°): Records minor natural variation
- **Too loose** (> 10°): Misses subtle rotations

### 3. Wraparound Behavior
Test boundary cases:
```
359° → 1°:   Should calculate as 2° change
0° → 359°:   Should calculate as 1° change
180° → 0°:   Should calculate as 180° change (not ambiguous)
```

### 4. Multi-Hand Tracking
Verify left/right independence:
```
Rotate left palm while keeping right still:
  - left: [recorded angle changes]
  - right: [] (empty, no significant changes)
```

---

## Performance Notes

- **Computation**: O(1) per hand per frame - negligible overhead
- **Memory**: ~100 bytes per tracked angle per hand
- **Storage**: ~8 bytes per angle value (float), negligible for typical gestures

---

## Future Enhancements

### Optional - Dataset Integration
Include palm angle data in exported datasets:
```python
sample = {
    "voxel_hits": [...],
    "chain_code": [...],
    "palm_angle_changes": {
        "left": [...],
        "right": [...]
    }
}
```

### Optional - Visualization
Display current palm angle on captured frame:
```
[Right Palm: 45.2°] [Left Palm: 123.5°]
```

### Optional - Velocity Calculation
Track palm rotation velocity:
```python
angle_velocity = angle_change / frame_duration  # degrees per second
```

---

## Troubleshooting

### No Angles Recorded
- **Check**: Are you capturing with hand landmarks valid? (all 21 points)
- **Check**: Is palm angle threshold set too high? (try 2.0 for testing)
- **Check**: Are hands rotating > 5°? (many static poses won't have changes)

### Angles Seem Wrong
- **Verify**: Hand landmarks are normalized correctly
- **Verify**: Wrist and MCP landmarks are being used (indices 0, 5, 9, 13, 17)
- **Debug**: Print `palm_angle_info['angle_deg']` to verify values in 0-180 range

### Memory Growing
- **Check**: Are you calling `reset_hit_tracking()` between captures?
- **Check**: Should be resetting palm angle dicts automatically

---

## References

- **MediaPipe Hand Landmark Indices**: See [face_grid_3d.py#L20+](../face_grid_3d.py#L20)
- **Palm Angle Methods**: [face_grid_3d.py#L480-L650](../face_grid_3d.py#L480) (approximately)
- **Integration Point**: [face_grid_3d.py#L1160-L1170](../face_grid_3d.py#L1160)
- **Architecture Guide**: [docs/ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md)

