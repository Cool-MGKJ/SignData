# API Reference — SignData (Core functions)

This document summarizes the main classes and functions useful for developers.

## `HandCapture` (capture.py)
- `initialize()` — open camera and init MediaPipe Hands
- `get_landmarks(frame)` — returns list of dicts: `{'hand': 'left'|'right', 'landmarks': [(x,y,z), ...]}`
- `start_grid_tracking()` / `stop_grid_tracking()` — control grid capture lifecycle
- `get_chain_code()` — returns last captured chain code list
- `get_hit_order()` — returns last captured hit order list

## `FaceGrid3D` (face_grid_3d.py)
- `process_frame(frame)` — detect face, estimate grid centers, returns True on success
- `update_hit_tracking(landmarks_list)` — compute palm trigger and update voxel hits
- `get_voxel_centers()` — returns list of (x,y,z) voxel centers
- `get_hit_order()` — ordered list of voxel indices hit during capture
- `get_chain_code()` — list of direction indices (0-25)

## `normalize_hand_data(landmarks, apply_rotation=True)` (normalization.py)
- Normalize a single hand to wrist-origin, scaled to wrist->middle MCP distance, optional rotation alignment.

## `DatasetManager` (dataset_io.py)
- `add_sample(label, normalized_points, hand, hit_order, chain_code)` — store sample and return id
- `save_as_csv(path)` / `save_as_json(path)` — export dataset

