# ASL Dataset Collection Tool

A Python application for building a dataset of American Sign Language (ASL) signs using MediaPipe hand tracking and a webcam. This tool allows you to capture hand landmarks, normalize them to a consistent 3D space, and save labeled samples for machine learning training.

## Features

- **Real-time hand tracking**: Uses MediaPipe Hands to detect and track hand landmarks in 3D space
- **Normalized coordinates**: Converts raw landmarks to a consistent 3D coordinate system
- **Labeled dataset collection**: Capture samples with custom labels for each ASL sign
- **Multiple export formats**: Save datasets as CSV or JSON
- **User-friendly interface**: Simple desktop UI with camera preview and sample management

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
   - The camera preview will open showing your webcam feed with hand landmarks overlaid
   - Position your hands in front of the camera
   - Click **"Start Capture"** to begin recording
   - Make your ASL sign
   - Click **"Stop Capture"** to capture the current frame
   - Enter a label for the sign in the text field (e.g., "hello", "thank you")
   - Click **"Save Sample"** to add the sample to your dataset
   - Repeat for additional signs

3. **Viewing collected samples**:
   - The "Recent Sample" panel shows the normalized 3D coordinates of the last captured sample
   - The "Collected Samples" table shows all samples collected in the current session

4. **Exporting the dataset**:
   - Click **"Export Dataset"** to save all collected samples to a file
   - Choose between JSON or CSV format
   - The file will contain all samples with their labels and normalized coordinates

5. **Clearing the session**:
   - Click **"Clear Session"** to remove all samples from the current session (after confirmation)

## Dataset Format

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

2. **Normalization**: Raw landmarks are normalized to a consistent coordinate system:
   - Translated to origin (centered)
   - Scaled to unit size
   - Applied spacing ratio for overall scaling
   - Results in a fixed-size list of 3D points

3. **Data Storage**: Each sample includes:
   - Unique ID
   - User-provided label
   - Hand information (left/right/both)
   - Flattened 3D coordinates (x1, y1, z1, x2, y2, z2, ...)

4. **Export**: Samples can be exported in ML-friendly formats (CSV or JSON) for training models

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

