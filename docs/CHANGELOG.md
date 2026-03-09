# Changelog & Project History

This document chronicles the historical context of the SignData project, consolidating major architectural phases and recent code changes to help AI agents and developers understand *why* certain decisions were made.

---

## Evolution Phases (Jan 1-14, 2026)

### Phase 1: Foundation (Jan 1-2)
- Initialized repository, MediaPipe, OpenCV, Tkinter UI.

### Phase 2: Core 3D Grid System (Jan 3-5)
- Implemented the 160 voxel (8×10×2) grid.
- Defined face-centered coordinate system. 
- Formulated hit detection logic (Euclidean distance).

### Phase 3 & 4: Detection & Data IO (Jan 6-9)
- Integrated MediaPipe Hands/Pose.
- Created `dataset_io.py` for JSON/CSV storage.

### Phase 5 & 6: Data Normalization & Chain Codes (Jan 10-11)
- Implemented wrist-centric data scaling.
- Introduced 26-directional chain code vectors to encode trajectory movement independently of hand speed.

### Phase 7 & 8: Depth Scaling & Layer Triggers (Jan 12-13)
- Implemented **Dynamic Depth Scaling**.
- **Crucial Decision:** Scaling uses shoulder width for distance invariance. 
- **Crucial Decision:** Layer 0 is face-side (GREEN), Layer 1 is camera-side (RED).

### Phase 10: Palm Angle & Distance Tracking (Jan 14)
- Added capability to track (yaw, pitch, roll) and Euclidean hand-to-nose distance to distinguish different types of gestures that hit the same voxels.

---

## Recent Significant Code Changes

### 1. Palm Angle Tracking
**Files:** `face_grid_3d.py`
To capture hand orientation without flooding the dataset with noise, a 5° change threshold was instituted. 
- Uses wrist, middle MCP, pinky MCP, and thumb CMC to extract a palm plane normal.
- **Bug Fix:** A polymorphic helper `get_coords(landmark)` was added to gracefully handle both MediapPe landmark objects (with `.x`, `.y` attributes) and raw tuples, which initially broke the right hand angle gathering.

### 2. Distance Tracking
**Files:** `face_grid_3d.py`
Captures hand-to-nose distance.
- Uses a threshold of 0.05 normalized units to record changes. This enables detecting "feathering" or gestures opening towards/away from the face.

### 3. State Management Overhaul
**Files:** `face_grid_3d.py`, `dataset_io.py`, `main.py`
- With angles and distance, state variables (e.g., `palm_angles_left`, `trigger_distance_right`) were added securely to the JSON metadata export format in `add_sample()`.

---

## Key Historical Decisions (Do Not Alter)

1. **2-Layer Grid vs N-Layer Grid:** Chosen for computational simplicity vs expressiveness.
2. **26-Directional Chain Code:** Chosen over 8-directional to properly encapsulate 3D lattice moves.
3. **Face-Centered Coordinates:** Chosen as an invariant over the camera center because gestures relate biologically to the signer's body posture, not the camera's fixed position.
