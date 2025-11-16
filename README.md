# ASL Dataset Collection Tool

A Python application for building a dataset of American Sign Language (ASL) signs using MediaPipe hand tracking and a webcam. This tool allows you to capture hand landmarks, normalize them to a consistent 3D space, and save labeled samples for machine learning training.

## Features

- **Real-time hand tracking**: Uses MediaPipe Hands to detect and track hand landmarks in 3D space
- **Face-centered 3D grid tracking**: Uses MediaPipe Face Mesh to create a 3D volumetric grid around the face and track which grid points are hit by hand movements
- **Dual data representation**: Each sample includes both:
  - **Conventional hand-shape data**: Normalized 3D hand landmarks (shape of the hand)
  - **Hit-grid data**: Binary vector indicating which grid points were hit during the capture session
- **Normalized coordinates**: Converts raw landmarks to a consistent 3D coordinate system
- **Labeled dataset collection**: Capture samples with custom labels for each ASL sign
- **Multiple export formats**: Save datasets as CSV or JSON
- **User-friendly interface**: Simple desktop UI with camera preview, grid visualization, and sample management

## Requirements

- Python 3.12 for mediapipe 
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
   - The camera preview will open showing your webcam feed with hand landmarks and a face-centered grid overlaid
   - Position yourself so your face is clearly visible (the grid is centered on your nose)
   - Position your hands in front of the camera
   - Click **"Start Capture"** to begin recording
   - Make your ASL sign (the grid will track which points your hands hit during the entire capture)
   - Click **"Stop Capture"** to end the capture session
   - Enter a label for the sign in the text field (e.g., "hello", "thank you")
   - Click **"Save Sample"** to add the sample to your dataset (includes both hand-shape and hit-grid data)
   - Repeat for additional signs

3. **Viewing collected samples**:
   - The "Recent Sample" panel shows:
     - The normalized 3D coordinates of the last captured sample
     - The number of grid points hit during the capture (e.g., "Hit grid points: 87 / 1260")
   - The "Collected Samples" table shows all samples collected in the current session
   - During capture, the grid is visualized on the camera preview:
     - Gray points: Grid points not yet hit
     - Green points: Grid points that have been hit by hand landmarks

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

### JSON Format

The JSON export creates a file with the following structure:

```json
{
  "metadata": {
    "total_samples": 10,
    "exported_at": "2024-01-15T10:30:00",
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
- `num_grid_points`: Total number of grid points (typically 1,260 = 12 × 15 × 7)
- `num_hit_points`: Number of grid points that were hit during capture
- `g_0`, `g_1`, `g_2`, ...: Binary values (0 or 1) for each grid point indicating if it was hit

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
   - Identifies nose center (stable reference point)
   - Calculates eye positions to determine grid spacing
   - Creates a 3D volumetric grid (12 points wide × 15 points tall × 7 points deep = 1,260 points) centered on the nose
   - Grid extends approximately 4 points outside the head region on each side
   - Grid is a true 3D volume in space, with points distributed in width (x), height (y), and depth (z) dimensions

3. **Grid Hit Tracking**: During capture (from Start to Stop):
   - For each frame, hand landmarks are compared to grid points
   - If a hand landmark is within a threshold distance of a grid point, that point is marked as "hit"
   - The hit status accumulates over the entire capture session (not just a single frame)
   - Results in a binary vector: 1 if point was hit at least once, 0 otherwise

4. **Normalization**: Raw hand landmarks are normalized to a consistent coordinate system:
   - Translated to origin (centered)
   - Scaled to unit size
   - Applied spacing ratio for overall scaling
   - Results in a fixed-size list of 3D points

5. **Data Storage**: Each sample includes:
   - Unique ID
   - User-provided label
   - Hand information (left/right/both)
   - Flattened 3D coordinates (x1, y1, z1, x2, y2, z2, ...) - conventional hand-shape representation
   - Hit-grid vector (1,260 binary values) - which grid points were hit during capture

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
- The face-centered grid adapts to each person's face size and position, ensuring consistent spatial reference
- Grid visualization helps users understand which areas their hands are interacting with during sign capture
- The hit-grid method captures temporal information (movement patterns) while the hand-shape method captures static pose information

## License

This project is provided as-is for educational and research purposes.

