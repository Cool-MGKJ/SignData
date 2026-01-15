# Core Concepts: Algorithms and Technical Details

Deep dive into the mathematical formulas, algorithms, and concepts that power SignData.

## Dynamic Depth Scaling Algorithm

### Problem Statement
How do we automatically determine whether a hand is near the face (Layer 0) or extended toward the camera (Layer 1)?

### The Formula

```
relative_z = (hand_z - face_z_reference) / reference_length
```

**Terms:**
- `hand_z` = Raw Z-coordinate from MediaPipe hand (negative = toward camera)
- `face_z_reference` = Nose tip Z from Face Mesh landmark 4
- `reference_length` = Shoulder-to-shoulder distance from Pose

**Layer Selection:**
```
if relative_z >= 0:
    trigger_layer = Layer 0 (GREEN, inner layer, near face)
else:
    trigger_layer = Layer 1 (RED, outer layer, near camera)
```

### Complete Algorithm (5 Steps)

**Step 1: Get Reference Length**
```python
left_shoulder = pose_landmarks.landmark[11]  # left shoulder
right_shoulder = pose_landmarks.landmark[12]  # right shoulder
reference_length = distance(left_shoulder, right_shoulder)
# Typical value: 0.15-0.25 (normalized)
```

**Step 2: Get Face Z Reference**
```python
nose = face_landmarks.landmark[4]  # nose tip
face_z_reference = nose.z
# This is the zero-point for relative depth
```

**Step 3: Get Hand Z**
```python
trigger_point = calculate_palm_trigger_point(hand_landmarks)
hand_z = trigger_point[2]  # z-component
```

**Step 4: Calculate Relative Z**
```python
relative_z = (hand_z - face_z_reference) / reference_length
# Example: (0.5 - 0.4) / 0.2 = 0.5 → Layer 0
# Example: (0.3 - 0.4) / 0.2 = -0.5 → Layer 1
```

**Step 5: Select Layer and Trigger**
```python
if relative_z >= 0:
    voxels_in_layer = [0:80]  # Layer 0 indices
    color = GREEN
else:
    voxels_in_layer = [80:160]  # Layer 1 indices
    color = RED

# Find nearest voxel in selected layer
nearest_voxel = find_nearest(hand_trigger, voxels_in_layer)
if distance(nearest_voxel, hand_trigger) <= hit_radius:
    mark_hit(nearest_voxel)
```

### Why This Works: Key Properties

**1. Distance-Invariant**
- Normalizes by person's shoulder width
- Taller people don't need different thresholds
- Shorter people don't trigger layers prematurely

**2. Adaptive**
- Uses current face position (not fixed)
- Works if person moves closer/farther from camera
- Adjusts to head tilt automatically

**3. Robust**
- Single threshold (0) is simple and stable
- Works with tremors and involuntary motion
- Handles different gesture speeds

**4. Intuitive**
- Positive relative_z = hand toward face = inner layer
- Negative relative_z = hand toward camera = outer layer
- Maps naturally to hand motion

### Example Walkthrough

**Scenario:** Person performing gesture, 6 feet from camera

```
Setup:
  - Shoulder distance = 0.18 (normalized)
  - Nose Z = 0.45 (MediaPipe depth)
  
Frame 1: Hand approaching face
  - Hand Z = 0.48 (moved forward)
  - relative_z = (0.48 - 0.45) / 0.18 = 0.167
  - Since 0.167 >= 0 → Layer 0 (GREEN)
  
Frame 2: Hand extended toward camera
  - Hand Z = 0.35 (moved back)
  - relative_z = (0.35 - 0.45) / 0.18 = -0.556
  - Since -0.556 < 0 → Layer 1 (RED)
  
Frame 3: Back to face
  - Hand Z = 0.44 (moved forward again)
  - relative_z = (0.44 - 0.45) / 0.18 = -0.056
  - Since -0.056 < 0 → Layer 1 (RED) [still extended]
  
Frame 4: Closer to face
  - Hand Z = 0.47 (moved forward more)
  - relative_z = (0.47 - 0.45) / 0.18 = 0.111
  - Since 0.111 >= 0 → Layer 0 (GREEN) [switched]
```

---

## Layer Triggering Logic

### Hit Detection Process

For each hand in each frame:

1. **Calculate trigger point** (palm center)
   ```python
   wrist = landmarks[0]
   fingers = [landmarks[5], landmarks[9], landmarks[17]]
   trigger_point = weighted_average(wrist, fingers)
   ```

2. **Get hand depth** (from trigger point)
   ```python
   hand_z = trigger_point.z
   ```

3. **Calculate relative Z** (depth scaling)
   ```python
   relative_z = (hand_z - face_z_reference) / reference_length
   ```

4. **Determine layer**
   ```python
   layer = 0 if relative_z >= 0 else 1
   voxel_range = [0:80] if layer == 0 else [80:160]
   ```

5. **Find nearest voxel** (2D search in layer)
   ```python
   min_distance = infinity
   nearest_voxel = -1
   for voxel_idx in voxel_range:
       voxel_x, voxel_y, voxel_z = voxel_centers[voxel_idx]
       dist = sqrt((hand_x - voxel_x)² + (hand_y - voxel_y)²)
       if dist < min_distance:
           min_distance = dist
           nearest_voxel = voxel_idx
   ```

6. **Check hit radius** (trigger if close enough)
   ```python
   hit_radius = 0.12  # normalized units
   if min_distance <= hit_radius and nearest_voxel >= 0:
       mark_hit(nearest_voxel)
       update_chain_code(nearest_voxel)
   ```

### Why 2D Search?

Layer already determined by Z → Only search X, Y within that layer
- **Reason 1:** Faster (search 80 voxels instead of 160)
- **Reason 2:** More accurate (Z already matched)
- **Reason 3:** Prevents cross-layer false hits

---

## 26-Directional Chain Code

### Purpose
Encode the direction of hand movement as it travels through voxels, creating a trajectory signature.

### Direction Vectors

26 directions in 3D space (face-centered coordinate system):

```
Directions: All combinations of {-1, 0, 1} for {x, y, z} except (0,0,0)
Total: 3³ - 1 = 26 directions

Visual (XY plane at z=0):
    6  7  8     (back-left, back, back-right)
    3  .  5     (left, center, right)
    0  1  2     (forward-left, forward, forward-right)

Add 9 more for z != 0:
    9-17:   forward-back-up variations
    18-25:  forward-back-down variations
```

### Calculation Method

For each voxel hit:

1. **Calculate direction vector**
   ```python
   prev_voxel = voxel_centers[voxel_hit_order[-2]]
   curr_voxel = voxel_centers[voxel_hit_order[-1]]
   direction_vector = curr_voxel - prev_voxel
   ```

2. **Find closest match** (cosine similarity)
   ```python
   max_similarity = -1
   best_direction_idx = 0
   
   for idx, direction in enumerate(DIRECTION_26):
       # Normalize both vectors
       norm_dir = direction_vector / ||direction_vector||
       norm_ref = direction / ||direction||
       
       # Cosine similarity = dot product (for normalized vectors)
       similarity = dot_product(norm_dir, norm_ref)
       
       if similarity > max_similarity:
           max_similarity = similarity
           best_direction_idx = idx
   ```

3. **Append to chain code**
   ```python
   chain_code.append(best_direction_idx)
   ```

### Example

```
Gesture: Hand moves right and down through voxels
  
  Voxel Sequence: [31, 39, 47, 55, ...]
  
  Direction 1 (31→39): (0.1, 0.0, 0.01) → Right ≈ Direction 5
  Direction 2 (39→47): (0.05, 0.1, -0.01) → Right-Down ≈ Direction 2
  Direction 3 (47→55): (0.0, 0.1, 0.0) → Down ≈ Direction 1
  
  Chain Code: [5, 2, 1, ...]
```

### Uses

- **Trajectory Validation** - Check if gesture path is plausible
- **Gesture Recognition** - Chain code is a gesture signature
- **Post-Processing** - Detect and filter unintended movements
- **Data Compression** - Encode movement pattern concisely

---

## Palm Angle Tracking (Yaw/Pitch/Roll)

### Purpose
Capture hand orientation to distinguish gestures that differ in hand position but same voxel hits.

### Algorithm

**Step 1: Extract Key Landmarks**
```python
wrist = hand_landmarks[0]      # origin
index_mcp = hand_landmarks[5]  # finger base
middle_mcp = hand_landmarks[9]
pinky_mcp = hand_landmarks[17]
```

**Step 2: Create Direction Vectors**
```python
v_index = index_mcp - wrist
v_middle = middle_mcp - wrist
v_pinky = pinky_mcp - wrist

# Normalize
v_index_norm = v_index / ||v_index||
v_middle_norm = v_middle / ||v_middle||
```

**Step 3: Calculate Palm Orientation Vectors**
```python
# Palm normal (perpendicular to palm)
palm_normal = cross_product(v_index_norm, v_middle_norm)
palm_normal = palm_normal / ||palm_normal||

# Palm forward (average finger direction)
palm_forward = (v_index_norm + v_middle_norm) / 2
palm_forward = palm_forward / ||palm_forward||

# Palm right (perpendicular to forward and normal)
palm_right = cross_product(palm_forward, palm_normal)
palm_right = palm_right / ||palm_right||

# Palm up (negative normal)
palm_up = -palm_normal
```

**Step 4: Extract Euler Angles**
```python
# Yaw (rotation around vertical axis)
yaw = arctan2(palm_right[1], palm_right[0])
yaw_degrees = degrees(yaw)

# Pitch (rotation looking up/down)
pitch = arcsin(clamp(-palm_forward[2], -1, 1))
pitch_degrees = degrees(pitch)

# Roll (twist around forward axis)
roll = arctan2(palm_up[0], palm_up[1])
roll_degrees = degrees(roll)

return (yaw_degrees, pitch_degrees, roll_degrees)
```

### Threshold-Based Recording

Angles recorded only when change exceeds 5 degrees:
```python
initial_angles = calculate_palm_angles(landmarks)  # Frame 1

for each subsequent frame:
    current_angles = calculate_palm_angles(landmarks)
    
    yaw_change = abs(current[0] - last_recorded[0])
    pitch_change = abs(current[1] - last_recorded[1])
    roll_change = abs(current[2] - last_recorded[2])
    
    if any([yaw_change > 5, pitch_change > 5, roll_change > 5]):
        angles_array.append(current_angles)
        last_recorded = current_angles
```

### Why Threshold?

- **Noise Reduction** - Ignores tremors and natural variation
- **Efficiency** - Only records significant changes
- **Robustness** - Handles different capture frame rates

---

## Distance Tracking

### Purpose
Capture hand motion toward/away from face to detect different hand opening/closing patterns.

### Calculation

For each hand, calculate distance from trigger point to nose:

```python
def calculate_distance(trigger_point, nose_position):
    # Both in normalized coordinates [0, 1]
    dx = nose_position[0] - trigger_point[0]
    dy = nose_position[1] - trigger_point[1]
    dz = nose_position[2] - trigger_point[2]
    
    distance = sqrt(dx² + dy² + dz²)
    return distance
```

**Example:**
```
Frame 1:
  - Nose: (0.5, 0.5, 0.45)
  - Hand: (0.5, 0.5, 0.60)
  - Distance: sqrt(0² + 0² + 0.15²) = 0.15

Frame 2:
  - Nose: (0.5, 0.5, 0.45)
  - Hand: (0.5, 0.5, 0.50)
  - Distance: sqrt(0² + 0² + 0.05²) = 0.05

Distance change: 0.15 - 0.05 = 0.10 > 0.05 threshold → Record
```

### Threshold-Based Recording

Distance recorded only when change exceeds 0.05 units:
```python
initial_distance = calculate_trigger_distance(trigger_point, nose)

for each subsequent frame:
    current_distance = calculate_trigger_distance(trigger_point, nose)
    
    distance_change = abs(current_distance - last_recorded_distance)
    
    if distance_change > 0.05:
        distance_array.append(current_distance)
        last_recorded_distance = current_distance
```

### Interpretation

```
Large distance decreases → Hand approaching face
                        → Opening hand toward face
                        → Potential feather/contact gesture

Large distance increases → Hand moving away from face
                        → Opening hand away from face
                        → Potential throw/release gesture
```

---

## Head Orientation (Yaw and Pitch)

### Yaw Calculation (Left-Right Turn)

**Using Eye Positions:**
```python
left_eye = face_landmarks.landmark[133]   # left eye inner
right_eye = face_landmarks.landmark[33]   # right eye outer

# 3D vector from left to right eye
eye_vector = (
    right_eye.x - left_eye.x,
    right_eye.y - left_eye.y,
    right_eye.z - left_eye.z
)

# Yaw angle from horizontal plane
yaw_rad = arctan2(eye_vector[2], eye_vector[0])

# Clamp to ±40 degrees
yaw_rad = clamp(yaw_rad, -40°, +40°)
yaw_degrees = degrees(yaw_rad)
```

### Pitch Calculation (Up-Down Nod)

**Using Forehead-Chin Vector:**
```python
forehead = face_landmarks.landmark[10]    # between eyebrows
chin = face_landmarks.landmark[152]       # chin center

# 3D vector from forehead to chin
vertical_vector = (
    chin.x - forehead.x,
    chin.y - forehead.y,
    chin.z - forehead.z
)

# Pitch angle from vertical
pitch_rad = arctan2(vertical_vector[2], vertical_vector[1])

# Invert to match intuition (nodding down = positive pitch)
pitch_rad = -pitch_rad

# Clamp to ±30 degrees
pitch_rad = clamp(pitch_rad, -30°, +30°)
pitch_degrees = degrees(pitch_rad)
```

### Grid Rotation Application

The calculated yaw and pitch are applied to rotate all voxel positions:

```python
# Yaw rotation (around Y-axis)
rotated_x = x * cos(yaw) + z * sin(yaw)
rotated_z = -x * sin(yaw) + z * cos(yaw)

# Pitch rotation (around X-axis)
rotated_y = y * cos(pitch) - rotated_z * sin(pitch)
final_z = y * sin(pitch) + rotated_z * cos(pitch)
```

This ensures the grid always "faces" the hand, regardless of head orientation.

---

## Normalization

### Hand Landmark Normalization

**Step 1: Center on Wrist**
```python
wrist = landmarks[0]
centered_landmarks = [landmark - wrist for landmark in landmarks]
```

**Step 2: Calculate Scale**
```python
# Use average distance from wrist to fingers
distances = []
for finger_tip in [8, 12, 16, 20]:
    dist = ||landmarks[finger_tip] - wrist||
    distances.append(dist)
scale = mean(distances)  # Typically 0.1-0.2
```

**Step 3: Apply Scale Normalization**
```python
normalized = [landmark / scale for landmark in centered_landmarks]
```

**Step 4: Optional Rotation** (for ML training)
```python
if apply_rotation:
    # Calculate principal axes via PCA
    # Align hand to standard orientation
    # Apply rotation transformation
```

**Result:** All hand landmarks in [-1, 1] range, comparable across gestures

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Time |
|-----------|-----------|------|
| Face detection | O(1) | ~30ms |
| Hand detection | O(1) | ~30ms |
| Voxel calculation | O(W×H×D) | <1ms |
| Nearest voxel | O(L) | <1ms |
| Chain code | O(1) | <1ms |
| Palm angles | O(1) | <1ms |
| Distance calc | O(1) | <1ms |

Where: W=8, H=10, D=2, L=80 (voxels per layer)

### Convergence

- **Layer detection:** 1 frame (immediate feedback)
- **Chain code:** On each voxel hit
- **Palm angles:** Every frame, recorded when > 5° change
- **Distance:** Every frame, recorded when > 0.05 change

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| z | Z-coordinate (depth) |
| z_ref | Face Z-reference (nose) |
| L | Reference length (shoulder distance) |
| r_z | Relative Z (dimensionless) |
| θ_y | Yaw angle |
| θ_p | Pitch angle |
| θ_r | Roll angle |
| d | Distance (Euclidean) |
| v | Vector |
| \\|v\\| | Vector magnitude |
| ⊗ | Cross product |
| · | Dot product |

---

## Assumptions and Limitations

### Assumptions
1. MediaPipe landmark quality is reasonable
2. Face is visible during capture (for Z-reference)
3. At least one shoulder visible (for reference length)
4. Hand mostly in frame (for trigger point)

### Limitations
1. Z-depth is relative (not absolute metric distance)
2. Heavy occlusion breaks hand detection
3. Multiple faces confuse the system (uses first)
4. Very fast motion may skip voxels
5. Tremors can trigger false direction changes

### Mitigations
1. Threshold-based recording reduces noise
2. Last-known-posture fallback for occlusion
3. Hit radius provides tolerance for speed
4. Cosine similarity for direction matching is robust

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview and design
- [DEVELOPMENT.md](DEVELOPMENT.md) - Implementation in code
- [USER_GUIDE.md](USER_GUIDE.md) - How to use the system

---

**Version:** January 2026  
**Status:** Active
