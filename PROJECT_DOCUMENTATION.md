# SignData — Project Documentation

This document describes the SignData / ASL Dataset Collection Tool: architecture, core features, data formats, and how the main modules work. It is intended to help developers, IDEs, and AI agents understand and extend the codebase.

## Overview

SignData captures labeled ASL sign samples using MediaPipe hand & face tracking and a face-centered 3D voxel grid. Each sample contains:
- Normalized 3D hand landmarks (live-normalized during capture)
- Hit-grid data: ordered voxel indices visited during the capture
- Chain code: sequence of directional indices (0-25) representing movement trajectory

Primary goals:
- Capture clean, normalized examples for ML training
- Record both static pose (hand landmarks) and dynamic spatial path (voxel hits and chain code)
- Keep live normalization as the canonical data representation

## Project Structure (files)

- `main.py`: Application coordinator; wires UI, capture, normalization, and dataset manager.
- `capture.py`: `HandCapture` — webcam handling, MediaPipe Hands integration, forwards frames for grid updates.
- `face_grid_3d.py`: `FaceGrid3D` — constructs the face-centered voxel grid, handles hit tracking, path tracking, and chain-code generation.
- `normalization.py`: Functions to normalize hand landmarks during capture (`normalize_hand_data`).
- `dataset_io.py`: `DatasetManager` — in-memory storage, add sample, export to JSON/CSV.
- `ui.py`: Tkinter GUI, camera preview, controls, and sample table.
- `PROJECT_DOCUMENTATION.md`: (this file) project overview and developer guidance.

## High-level Flow

1. App starts (`main.py`) and initializes `HandCapture`, UI, and `DatasetManager`.
2. `HandCapture.initialize()` sets up the webcam and MediaPipe Hands.
3. Video loop: frames are read, `FaceGrid3D.process_frame()` updates grid centers and pose references.
4. When user clicks "Start Capture":
   - `HandCapture.start_grid_tracking()` resets hits and begins chain capture.
5. During capture: each frame calls `FaceGrid3D.update_hit_tracking()` which:
   - Computes a palm trigger point from hand landmarks (palm center ↔ fingertips interpolation)
   - Computes `relative_z` using face Z reference and a body reference length
   - Maps `relative_z` to one of two layers (Face Layer or Forward Layer)
   - Finds nearest voxel in that layer (2D proximity) and marks hits
   - Updates `voxel_hit_order` and appends direction indices to `current_gesture_chain` (chain code)
6. When user clicks "Stop Capture":
   - `HandCapture.stop_grid_tracking()` stops chain capture and preserves `voxel_hit_order` and `current_gesture_chain`
7. On "Save Sample":
   - `normalize_hand_data()` is applied to each hand and the flattened normalized points and metadata (hand label, hit order, chain_code) are stored in `DatasetManager`.

## Important Concepts

### 2-Layer Grid
- Grid size: default 8 (width) × 10 (height) × 2 (depth) = 160 voxels
- Hierarchy: **Camera → Layer 1 → Layer 0 → Face**
- **Layer 1** (z_idx=0, voxels 0–79): First layer as hand moves in. Hand extended toward camera yields `relative_z < 0`.
- **Layer 0** (z_idx=1, voxels 80–159): Second layer as hand approaches face. Hand at or behind face yields `relative_z >= 0`.
- Voxel indexing (linear): `idx = z_idx * (breadth * length) + y_idx * breadth + x_idx`.

### Palm Trigger Point
- Single point derived per hand per frame.
- Interpolates between palm base and fingertip average based on finger spread metric.
- Used to determine which voxel (if any) is hit.

### Chain Code (Directional Trajectory)
- 26 discrete directions corresponding to face/edge/corner unit vectors in 3D.
- When successive distinct voxels are hit, the direction between their centers is matched to the closest of the 26 directions using cosine similarity.
- Chain code is stored with each sample as a list of integers (0-25). It is not used for live validation.

### Normalization
- Performed live on each captured hand via `normalize_hand_data()`:
  - Wrist (index 0) moved to origin
  - Scaled by 3D distance wrist → middle finger MCP (index 9)
  - Optional rotation aligning wrist→middle-MCP to +Y axis
- No offline normalization script is used; live-normalized data is the canonical dataset representation.

## Data Format (per sample)

JSON sample structure (example):

{
  "id": 1,
  "label": "hello",
  "hand": "right",
  "points": [ { "index":0, "x":..., "y":..., "z":... }, ... ],
  "points_flat": [x1,y1,z1,x2,y2,z2,...],
  "num_points": 21,
  "timestamp": "...",
  "hit_order": [45, 46, 47, 80, ...],
  "chain_code": [2,5,14,14,...]
}

- `hit_order` records the sequence of voxel indices hit during capture.
- `chain_code` records the discrete movement directions between successive distinct voxel hits.

## Where To Start When Extending

- To change grid resolution: modify `breadth`, `length`, and `depth_layers` in `FaceGrid3D` constructor and update UI/text accordingly.
- To adjust hit sensitivity: tune `hit_radius_norm` in `FaceGrid3D`.
- To add advanced validation: implement post-processing scripts that consume `hit_order` + `chain_code` for clustering or matching.

## Notes for ML Engineers

- The dataset stores both static (normalized landmarks) and dynamic (hit_order, chain_code) signals; both can be used together in a model.
- Live normalization ensures consistency across samples captured in-session.
- Chain codes are suitable features for sequence-based models or for constructing edit-distance based similarity metrics.

## Quick Commands

Run the app:

```bash
pip install -r requirements.txt
python main.py
```

Export dataset from the UI using the Export button.

