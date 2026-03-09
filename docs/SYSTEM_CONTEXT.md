# System Context & Architecture

This document serves as the single source of truth for the core architecture, invariants, and implementation details of the SignData system. It is optimized for AI agents and human developers to understand the system without breaking previous functionality.

## 1. System Architecture & The Voxel Grid

SignData is an ASL gesture recognition dataset collection system that maps hand movements into a 3D, face-centered voxel grid.

### Key Statistics
- **Grid Size:** 160 voxels (8 wide × 10 tall × 2 deep)
- **Layers:** 2 (Layer 0 near face, Layer 1 near camera)
- **Input:** MediaPipe Hand (21 landmarks), Face Mesh (nose tip = origin), Pose (shoulders = reference length)
- **Output:** Normalzied coordinates, Voxel hit sequence, 26-direction chain code, Palm angles (yaw/pitch/roll), Hand-to-face distance

### Face-Centered Coordinate System Invariants
- **Origin:** The grid MUST center on the nose tip (Face Mesh landmark 4).
- **Rotation:** The grid MUST rotate with head yaw and pitch so it always "faces" the hand.
- **Normalization:** All coordinates are normalized to [0, 1] relative to the face and camera frame.

### Layer Layout
- **Layer 1 (RED):** Voxels 80-159. The outer layer, closer to the camera.
- **Layer 0 (GREEN):** Voxels 0-79. The inner layer, closer to the face.

---

## 2. Core Algorithms (Crucial Invariants)

ANY changes to the spatial tracking logic MUST respect these core algorithms to avoid corrupting the dataset schema.

### A. Dynamic Depth Scaling (Layer Triggering)
The system automatically adapts to different arm lengths and depths using this formula:
```python
relative_z = (hand_z - face_z_reference) / reference_length
```
- **Rule:** If `relative_z >= 0`, trigger Layer 0. If `relative_z < 0`, trigger Layer 1.
- `reference_length` is the distance between the left and right shoulders (Pose landmarks 11 and 12).
- The layer search is a **2D nearest-neighbor search** within the chosen layer.

### B. 26-Directional Chain Code
When a hand moves between voxels, the trajectory is encoded into one of 26 3D directions using cosine similarity.
- **Rule:** The chain code captures trajectory sequence independent of hand speed.

### C. Palm Angle Tracking
Hand orientation (yaw, pitch, roll) is extracted using cross products of the wrist, index MCP, middle MCP, and pinky MCP.
- **Rule (Thresholding):** To prevent noise, angles are ONLY recorded when the change from the last recorded frame exceeds **5 degrees**.

### D. Distance Tracking
The Euclidean distance from the palm trigger point to the nose.
- **Rule (Thresholding):** Distance is ONLY recorded when the change exceeds **0.05 normalized units**.

---

## 3. Codebook & API Structure

The application is structured into specific modules. When modifying the system, adhere to these responsibilities:

### `main.py`
Orchestrator. Handles the `ASLDataCollectionApp` class, state management for currently captured variables (hits, chain codes, angles, distances), and saving.

### `capture.py`
Handles MediaPipe pipelines. Provides `HandCapture` which exposes `get_landmarks()`, `get_hit_order()`, `get_chain_code()`, etc.

### `face_grid_3d.py`
The mathematical engine. Contains `FaceGrid3D`.
- Computes the 160 voxel centers based on face position.
- Calculates `relative_z` and processes hit tracking.
- Renders the grid visuals (green/red voxels, polylines).

### `dataset_io.py`
Handles `DatasetManager` for JSON and CSV exports. 
- **Critical File Schema:** Changing the arguments to `add_sample(label, normalized_points, hand, hit_order, chain_code, palm_angles_left, palm_angles_right, trigger_distance_left, trigger_distance_right)` will break existing datasets.

### `normalization.py`
Contains `normalize_hand_data()`. Translates raw landmarks to coordinate space centered on the wrist and scaled by average finger distance.

### `ui.py`
Tkinter GUI wrapper (`ASLDataCollectionUI`). Must invoke callbacks registered by `main.py`.

---

## 4. Performance Expectations
- Face/Hand detection: ~30ms each.
- Voxel tracking, chain code, and vector math: < 1ms per frame via NumPy optimization.
- **Total Loop Expectation:** ~10-15 FPS with visual rendering.
