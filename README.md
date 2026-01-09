# ASL Dataset Collection Tool

A Python application for building a dataset of American Sign Language (ASL) signs using MediaPipe hand tracking and a webcam. This tool allows you to capture hand landmarks, normalize them to a consistent 3D space, and save labeled samples for machine learning training.

## Features

- **Real-time hand tracking**: Uses MediaPipe Hands to detect and track hand landmarks in 3D space
- **Face-centered 3D grid tracking**: Uses MediaPipe Face Mesh to create a 3D volumetric grid around the face and track which grid points are hit by hand movements
- **MiDaS monocular depth estimation**: Optional depth mode using MiDaS models for relative depth estimation without hardware depth cameras
- **Dual data representation**: Each sample includes both:
  - **Conventional hand-shape data**: Normalized 3D hand landmarks (shape of the hand)
  - **Hit-grid data**: Binary vector indicating which grid points were hit during the capture session
- **Normalized coordinates**: Converts raw landmarks to a consistent 3D coordinate system
- **Labeled dataset collection**: Capture samples with custom labels for each ASL sign
- **Multiple export formats**: Save datasets as CSV or JSON
- **User-friendly interface**: Simple desktop UI with camera preview and sample management

## Requirements

- Python 3.8+ for mediapipe 
- Webcam/camera connected to your computer
- Windows, macOS, or Linux

## Installation

1. **Clone or download this project** to your local machine.

2. **Navigate to the project directory**:
   ```bash
   cd signTalk
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   python main.py
   ```
## Usage

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **Using the application**:
   - The camera preview will open showing your webcam feed with hand landmarks overlaid
   - Position your hands in front of the camera
   - Click **"Start Capture"** to begin recording
   - Make your ASL sign
   - Click **"Stop Capture"** to capture the current frame
   - Enter a label for the sign in the text field (e.g., "hello", "thank you")
   - Click **"Save Sample"** to add the sample to your dataset
   - Repeat for additional signs

3. **Viewing collected samples**:
   - The "Recent Sample" panel shows:
     - The normalized 3D coordinates of the last captured sample
     - The number of grid points hit during the capture (e.g., "Hit grid points: 87 / 1260" - always 3D voxel grid)
   - The "Collected Samples" table shows all samples collected in the current session
   - During capture, the grid is visualized on the camera preview:
     - Gray points: Grid points not yet hit
     - Green points: Grid points that have been hit by hand landmarks
   - **Depth Mode indicator**: Shows "MiDaS: Active" (green) when MiDaS depth is working, or "MiDaS: Offline" (gray) when disabled or unavailable
   - **Hit Count display**: Shows the number of voxels hit (X/1260 - always 3D voxel grid)

4. **Exporting the dataset**:
   - Click **"Export Dataset"** to save all collected samples to a file
   - Choose between JSON or CSV format
   - The file will contain all samples with their labels and normalized coordinates

5. **Clearing the session**:
   - Click **"Clear Session"** to remove all samples from the current session (after confirmation)

## Dataset Format

Each sample in the dataset contains two complementary data representations:

1. **Conventional hand-shape data**: Normalized 3D hand landmarks representing the shape of the hand
2. **Hit-grid data**: A binary vector (0s and 1s) indicating which grid points were hit during the capture session

**Note**: The grid is always 3D (12 × 15 × 7 = 1,260 voxels). When using MiDaS depth mode, the z-coordinate comes from MiDaS depth estimation. When depth mode is "off", the system uses MediaPipe landmark z values as fallback (normalized to 0-1 range).

### JSON Format

The JSON export creates a file with the following structure:

```json
{
  "metadata": {
    "total_samples": 10,
    "exported_at": "2024-01-10T10:30:00",
    "format_version": "1.0"
  },
  "samples": [
    {
      "id": 1,
      "label": "hello",
      "hand": "right",
      "num_points": 21,
      "points": [x1, y1, z1, x2, y2, z2, ...],
      "hit_grid": [0, 1, 1, 0, 1, ...],
      "num_grid_points": 1260,
      "num_hit_points": 87,
      "hit_grid_midas": [0, 1, 0, 1, ...],
      "depth_mode": "midas",
      "grid_center_z": 0.523,
      "grid_spacing_norm": 0.045,
      "hit_points_midas": [x1, y1, z1, x2, y2, z2, ...],
      "hit_points_midas_count": 23,
      "timestamp": "2024-01-15T10:25:00"
    },
    ...
  ]
}
```

### CSV Format

The CSV export creates a file with columns:
- `id`: Unique sample ID
- `label`: The ASL sign label
- `hand`: Which hand(s) detected ("left", "right", "both", "unknown")
- `num_points`: Number of 3D points in the sample
- `point_0_x`, `point_0_y`, `point_0_z`, `point_1_x`, ...: Flattened 3D coordinates
- `num_grid_points`: Total number of voxels (always 1,260 = 12 × 15 × 7)
- `num_hit_points`: Number of grid points that were hit during capture
- `g_0`, `g_1`, `g_2`, ...: Binary values (0 or 1) for each grid point indicating if it was hit (backward compatibility)
- **3D Voxel Grid Fields**:
  - `hit_grid_3d`: Binary vector (1,260 values) for 3D voxel hits (z-major ordering)
  - `voxel_paths`: JSON object mapping landmark names to voxel index arrays
  - `grid_dims`: [12, 15, 7] - voxel grid dimensions
  - `depth_mode`: "midas" or "off"
  - `grid_center_z`: Depth value at the nose pixel (0.0-1.0, MiDaS or MediaPipe z)
  - `grid_spacing_norm`: Normalized grid spacing used
  - `hit_points_midas`: Flattened 3D coordinates of hit voxel centers [x1, y1, z1, x2, y2, z2, ...]
  - `hit_points_midas_count`: Number of hit voxels

## Project Structure

```
signTalk/
├── main.py              # Application entry point
├── capture.py           # Webcam and MediaPipe handling
├── normalization.py     # Landmark normalization functions
├── dataset_io.py        # Dataset storage and export
├── ui.py                # User interface (tkinter)
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── data/               # Auto-created directory for dataset files
    └── asl_dataset.json # Auto-saved samples (JSON format)
```

## How It Works

1. **Hand Detection**: MediaPipe Hands processes each camera frame to detect hand landmarks (21 points per hand in 3D space)

2. **Face Detection and Grid Construction**: MediaPipe Face Mesh detects the face and constructs a 3D volumetric grid:
   - Always creates a 3D voxel grid (12 points wide × 15 points tall × 7 points deep = 1,260 voxels) centered on the nose
   - **Depth mode "off"**: Uses MediaPipe landmark z values (normalized to 0-1 range) for depth
   - **Depth mode "midas"**: Uses MiDaS monocular depth estimation for more accurate depth:
     - Identifies nose center (stable reference point)
     - Calculates eye positions to determine grid spacing
     - Uses MiDaS monocular depth estimation to get relative depth at each grid point
     - Grid spacing is scaled by relative depth to keep grid stable across distances
     - Depth values are normalized to 0.0-1.0 range
   - Grid extends around the head region
   - Voxel centers are distributed in width (x), height (y), and depth (z) dimensions
   - **Voxel ordering**: z-major (layer by layer from near→far), within each layer: row-major over Y then X
     - Index formula: `idx = z_idx * (12 * 15) + y_idx * 12 + x_idx`

3. **Voxel Hit Tracking**: During capture (from Start to Stop):
   - For each frame, hand landmarks are compared to voxel centers in 3D space
   - **Depth mode "off"**: Uses MediaPipe landmark z values (normalized) for 3D distance calculation
   - **Depth mode "midas"**: Uses MiDaS depth at hand landmark pixel locations for 3D distance calculation
   - If a hand landmark is within `hit_radius_norm` (default: 0.035) of a voxel center (in normalized x, y, z space), that voxel is marked as "hit"
   - The hit status accumulates over the entire capture session (not just a single frame)
   - Results in a binary vector `hit_grid_3d`: 1 if voxel was hit at least once, 0 otherwise
   - **Voxel paths**: For tracked landmarks (default: index_tip, thumb_tip), records ordered sequence of voxel indices visited (only when voxel changes, no duplicates)

4. **Normalization**: Raw hand landmarks are normalized to a consistent coordinate system:
   - Translated to origin (centered)
   - Scaled to unit size
   - Applied spacing ratio for overall scaling
   - Results in a fixed-size list of 3D points

3. **Data Storage**: Each sample includes:
   - Unique ID
   - User-provided label
   - Hand information (left/right/both)
   - Flattened 3D coordinates (x1, y1, z1, x2, y2, z2, ...) - conventional hand-shape representation (always preserved, unchanged)
   - **3D Voxel Grid Fields**:
     - `hit_grid_3d`: 1,260-length binary vector (z-major ordering) - which voxels were hit during capture
     - `voxel_paths`: Dictionary mapping landmark names to ordered lists of voxel indices (e.g., `{"index_tip": [45, 67, 89, ...]}`)
     - `grid_dims`: [12, 15, 7] - voxel grid dimensions [breadth, length, depth_layers]
     - `depth_mode`: "midas" or "off"
     - `grid_center_z`: Depth at nose pixel (MiDaS or MediaPipe z, normalized 0-1)
     - `grid_spacing_norm`: Normalized spacing used
   - **Backward compatibility fields** (deprecated but preserved):
     - `hit_grid_midas`: Same as `hit_grid_3d` (for compatibility)
     - `hit_points_midas`: 3D coordinates of hit voxel centers

6. **Export**: Samples can be exported in ML-friendly formats (CSV or JSON) for training models

### Why Two Data Representations?

- **Hand-shape data**: Captures the static shape/pose of the hand at the end of the capture
- **Hit-grid data**: Captures the dynamic spatial pattern of hand movement relative to the face during the entire capture session

Together, these provide complementary information that can improve ASL recognition accuracy.

## Configuration

You can modify normalization parameters in `main.py`:

- `num_points_per_hand`: Number of landmarks per hand (default: 21, MediaPipe standard)
- `spacing_ratio`: Scaling factor for normalized coordinates (default: 1.0)

## Troubleshooting

- **Camera not detected**: Ensure your webcam is connected and not being used by another application
- **No hands detected**: Make sure your hands are clearly visible in the camera frame with good lighting
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

## Notes

- Samples are automatically saved to `data/asl_dataset.json` after each "Save Sample" action
- The application supports detecting up to 2 hands simultaneously
- Normalized coordinates are relative and scaled, making them suitable for ML training regardless of camera distance or hand size

## License

This project is provided as-is for educational and research purposes.

