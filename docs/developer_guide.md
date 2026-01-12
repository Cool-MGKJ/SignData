# Developer Guide — SignData

This guide explains the codebase structure, core concepts, and how to extend the project.

## Project Overview
SignData captures labeled ASL signs using MediaPipe Hands and a face-centered voxel grid. It stores each sample with live-normalized hand landmarks, voxel hit sequences, and a chain-code trajectory.

## Where to Start
1. Read `PROJECT_DOCUMENTATION.md` for a high-level overview and data examples.
2. Run the application locally with:

```bash
pip install -r requirements.txt
python main.py
```

## Core Modules
- `main.py`: Application wiring and Tkinter callbacks. See `ASLDataCollectionApp` for UI integration.
- `capture.py`: `HandCapture` wraps MediaPipe Hands and forwards landmark detections to `FaceGrid3D`.
- `face_grid_3d.py`: Implements the voxel grid, trigger logic, hit ordering, and chain code generation.
- `normalization.py`: Live normalization utilities used when saving samples.
- `dataset_io.py`: In-memory dataset and export functions.
- `ui.py`: Tkinter UI. Keep presentation separate from logic.

## Extension Points
- Grid resolution: change `breadth`, `length`, `depth_layers` in `FaceGrid3D`.
- Hit sensitivity: change `hit_radius_norm` in `FaceGrid3D`.
- Reference length calculation: `calculate_reference_length_from_pose()` in `FaceGrid3D` can be adapted for other body metrics.

## Coding Style
- Keep processing logic out of UI code.
- Normalize data at the point of saving; downstream scripts should consume the saved normalized representation.

## Testing
- The codebase uses procedural I/O; create small test harnesses that simulate `HandCapture.get_landmarks()` inputs to exercise `FaceGrid3D.update_hit_tracking()` and `_update_chain_code()`.

## Troubleshooting
- If MediaPipe fails to detect landmarks, verify camera permissions and lighting.
- If layers don't trigger correctly, ensure the `face_z_reference` and `reference_length` are being set by the Pose detection (visible in logs).

