# SignData: ASL Gesture Recognition Dataset Collection Tool

A computer vision-based system for collecting American Sign Language (ASL) gesture data using MediaPipe hand/pose detection and a 3D voxel grid tracking system.

## Quick Links for AI Agents & Developers
Technical documentation is centralized in the `/docs/` folder to provide robust context:
- **[System Context & Architecture](docs/SYSTEM_CONTEXT.md)** - Explains the 2-layer voxel grid, dynamic depth scaling, chain codes, and codebase invariants.
- **[Changelog & History](docs/CHANGELOG.md)** - Evolutionary timeline and major code modifications.

---

## Installation & Setup

### Requirements
- Python 3.8+
- Webcam/camera connected to your computer
- Windows, macOS, or Linux

### Setup Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

The UI will launch with a camera preview showing real-time hand detection.

---

## Using the Application

### Capturing a Gesture

1. **Position yourself:** Stand in front of the camera with hands visible.
2. **Click "Start Capture" (or press SPACE):** Begin recording the gesture.
3. **Perform the sign:** Make the ASL sign you want to record.
4. **Click "Stop Capture" (or press SPACE):** End the capture.
5. **Enter a label:** Type the sign name (e.g., "hello", "thank you").
6. **Click "Save Sample":** Add the sample to your dataset.

### What Gets Recorded

Each sample captures:
- **Spatial Data:** 21 normalized hand landmarks per hand, and a Sequence (hit order) of voxel indices touched during gesture.
- **Movement Data:** 26-directional trajectory encoding (chain code).
- **Orientation & Proximity:** Palm angles (Yaw, pitch, roll) and distance from face, strictly recorded when thresholds are met (5° and 0.05 units).

---

## Grid Visualization Explained

The system projects a 160-voxel (8x10x2) grid centered on your face.

- **Layer 0 (GREEN)** (voxels 0-79): Inner layer, closest to face. Triggers when your hand approaches or touches your face.
- **Layer 1 (RED)** (voxels 80-159): Outer layer, closest to camera. Triggers when your hand is extended toward the camera.

The system scales dynamically based on your shoulder width, automatically determining which layer to activate!

---

## Exporting the Dataset

1. **Click "Export Dataset"** button in the UI.
2. **Choose format:**
   - **JSON** - Better for ML pipelines, preserves the complex structural data (array of arrays).
   - **CSV** - Compatible with spreadsheet software.
3. **Select file location** and confirm.

**Example JSON Output:**
```json
{
  "metadata": {
    "total_samples": 10,
    "exported_at": "2026-01-14T13:35:20"
  },
  "samples": [
    {
      "id": 1,
      "label": "hello",
      "hand": "right",
      "num_points": 21,
      "points": [{"index": 0, "x": 0.0, "y": 0.0, "z": 0.0}],
      "hit_order": [0, 5, 12],
      "chain_code": [0, 1, 2],
      "palm_angles_left": [[10.2, 5.3, -2.1]],
      "palm_angles_right": [],
      "trigger_distance_left": [],
      "trigger_distance_right": [0.45, 0.48],
      "timestamp": "2026-01-14T13:35:09"
    }
  ]
}
```

---

## Troubleshooting & Tips

- **Camera Not Detected:** Try another USB port or close other apps using the camera.
- **Hands Not Detected:** Improve lighting and keep hands fully in frame.
- **Grid Not Showing:** The grid only highlights hits during active capture.
- **Clearing Data:** "Clear Session" removes all current samples. **Be sure to export first!**
