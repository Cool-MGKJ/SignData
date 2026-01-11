# ASL Dataset Collection Tool

A Python application for building a dataset of American Sign Language (ASL) signs using MediaPipe hand tracking and a webcam. This tool allows you to capture hand landmarks, normalize them to a consistent 3D space, and save labeled samples for machine learning training.

## Features

- **Real-time hand tracking**: Uses MediaPipe Hands to detect and track hand landmarks in 3D space
- **Face-centered 3D grid tracking**: Uses MediaPipe Face Mesh to create a 3D volumetric grid around the face and track which grid points are hit by hand movements
- **MediaPipe depth tracking**: Uses MediaPipe's built-in relative z-depth for all 3D calculations
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
     - The number of grid points hit during the capture (e.g., "Hit grid points: 87 / 240" - 3D voxel grid)
   - The "Collected Samples" table shows all samples collected in the current session
   - During capture, the grid is visualized on the camera preview:
     - Gray points: Grid points not yet hit
     - Green points: Grid points that have been hit by hand movements
   - **Hit Count display**: Shows the number of voxels hit (X/240 - 3D voxel grid using MediaPipe z-depth)

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

**Note**: The grid is always 3D (8 × 10 × 3 = 240 voxels by default). All z-coordinates come from MediaPipe's relative depth values (normalized to 0-1 range).

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
      "hit_order": [0, 5, 12, 45, ...],
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
- `hit_order`: Ordered list of voxel indices representing the sequence in which grid points were hit during capture

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
   - Creates a 3D voxel grid (default: 8 points wide × 10 points tall × 3 points deep = 240 voxels) centered on the nose
   - Uses MediaPipe Face Mesh landmark z values (normalized to 0-1 range) for all depth calculations
   - Identifies nose center (stable reference point) and calculates eye positions to determine grid spacing
   - Grid extends around the head region
   - Voxel centers are distributed in width (x), height (y), and depth (z) dimensions
   - **Voxel ordering**: z-major (layer by layer from near→far), within each layer: row-major over Y then X
     - Index formula: `idx = z_idx * (breadth * length) + y_idx * breadth + x_idx`
     - Example for default 8×10×3 grid: `idx = z_idx * 80 + y_idx * 8 + x_idx`

3. **Voxel Hit Tracking**: During capture (from Start to Stop):
   - For each frame, hand movements are tracked using a calculated palm trigger point
   - Uses MediaPipe landmark z values (normalized) for all 3D distance calculations
   - The trigger point moves based on finger spread (more towards fingertips when fingers are open)
   - If the trigger point is within `hit_radius_norm` (default: 0.12) of a voxel center (in normalized x, y, z space), that voxel is marked as "hit"
   - The hit status accumulates over the entire capture session (not just a single frame)
   - Results in an ordered list `hit_order`: sequence of voxel indices visited during the capture

4. **Normalization**: Raw hand landmarks are normalized to a consistent coordinate system:
   - Translated to origin (centered)
   - Scaled to unit size
   - Applied spacing ratio for overall scaling
   - Results in a fixed-size list of 3D points

3. **Data Storage**: Each sample includes:
   - Unique ID
   - User-provided label
   - Hand information (left/right/both)
   - Normalized 3D coordinates as numbered points: `[{"index": 0, "x": ..., "y": ..., "z": ...}, ...]`
   - `points_flat`: Flattened 3D coordinates (x1, y1, z1, x2, y2, z2, ...) for backward compatibility
   - `hit_order`: Ordered list of voxel indices representing the sequence in which grid points were hit during capture
   - All z-coordinates use MediaPipe's relative depth values

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

