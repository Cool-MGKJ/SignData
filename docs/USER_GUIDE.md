# User Guide: Running SignData

This guide covers installation, usage, and dataset export for the SignData ASL collection tool.

## Installation

### Requirements
- Python 3.8+
- Webcam/camera connected to your computer
- Windows, macOS, or Linux

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SignData
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

The UI will launch with a camera preview showing real-time hand detection.

---

## Using the Application

### Main Interface

The application displays:
- **Live camera feed** - Shows hands with MediaPipe landmarks overlaid
- **Control buttons** - Start/Stop capture, Save, Export, Clear
- **Recent sample panel** - Shows the last captured gesture
- **Samples table** - Lists all samples collected in the current session

### Capturing a Gesture

1. **Position yourself:** Stand in front of the camera with hands visible
2. **Click "Start Capture":** Begin recording the gesture
3. **Perform the sign:** Make the ASL sign you want to record
4. **Click "Stop Capture":** End the capture
5. **Enter a label:** Type the sign name (e.g., "hello", "thank you")
6. **Click "Save Sample":** Add the sample to your dataset

### What Gets Recorded

Each sample captures:

**Spatial Data:**
- 21 normalized hand landmarks per hand
- Hit order: Sequence of voxel indices touched during gesture
- Grid visualization shows which points were hit (Green for Layer 0, Red for Layer 1)

**Movement Data:**
- Chain code: 26-directional trajectory encoding
- Palm angles: Yaw, pitch, roll for each hand
- Distance: Hand distance from face for each hand

**Display Info:**
- "Hit grid points: 87 / 160" - Shows voxels hit out of total
- Voxel indices are overlaid on the visualization
- Different colors indicate which layer was triggered

---

## Grid Visualization

### The 2-Layer System

The grid consists of 160 voxels arranged as:
- **Layer 0** (voxels 0-79): Inner layer, closest to face → GREEN when hit
- **Layer 1** (voxels 80-159): Outer layer, closest to camera → RED when hit

### How It Works

The system automatically determines which layer to use based on hand depth:

```
FACE
 |
 v
[GREEN LAYER 0] ← Hand approaching/touching face
 |
 | (Depth boundary)
 |
[RED LAYER 1] ← Hand extended toward camera
 |
 v
CAMERA
```

**Dynamic Switching:**
- If your hand moves toward your face → GREEN points activate
- If your hand moves toward the camera → RED points activate
- The system automatically switches based on depth!

### Grid Visualization Details

- **8 wide × 10 tall × 2 deep = 160 total voxels**
- **Gray points**: Unhit voxels
- **Green points**: Layer 0 hits (inner, near face)
- **Red points**: Layer 1 hits (outer, near camera)
- **Voxel numbers**: Overlaid indices (0-159) for reference
- **Hit path**: Line showing order of voxel touches

---

## Viewing Collected Samples

### Recent Sample Panel
Shows details of the last captured sample:
- Normalized 3D coordinates of hand landmarks
- Number of voxels hit during capture
- Hit count display (e.g., "87 / 160 voxels")

### Samples Table
Lists all samples in the current session with:
- Sample ID
- ASL sign label
- Hand(s) detected (left, right, or both)
- Number of normalized points captured

---

## Exporting the Dataset

### Export Steps

1. **Click "Export Dataset"** button
2. **Choose format:**
   - **JSON** - Better for ML pipelines, preserves structure
   - **CSV** - Compatible with spreadsheet software
3. **Select file location** - Where to save the dataset
4. **Confirm** - File will be saved with all samples

### Exported Data Format

**JSON Structure:**
```json
{
  "metadata": {
    "total_samples": 10,
    "exported_at": "2026-01-14T13:35:20.560948"
  },
  "samples": [
    {
      "id": 1,
      "label": "hello",
      "hand": "right",
      "num_points": 21,
      "points": [{"index": 0, "x": 0.0, "y": 0.0, "z": 0.0}, ...],
      "hit_order": [0, 5, 12, ...],
      "chain_code": [0, 1, 2, ...],
      "palm_angles_left": [[10.2, 5.3, -2.1], ...],
      "palm_angles_right": [],
      "trigger_distance_left": [],
      "trigger_distance_right": [0.45, 0.48, ...],
      "timestamp": "2026-01-14T13:35:09.881392"
    }
  ]
}
```

**CSV Structure:**
- Each row is one sample
- Columns: ID, Label, Hand, Points (x₁, y₁, z₁, x₂, y₂, z₂, ...), Hit Order, Chain Code
- Angle and distance data stored as comma-separated values

---

## Data Collected Per Sample

### Normalized Hand Coordinates
- 21 landmarks per hand in 3D space
- Wrist-centered normalization
- Unit scale (0-1 range)
- Optional rotation normalization

### Hit Grid Information
- **hit_order**: Ordered voxel indices showing sequence of touches
  - Example: `[31, 23, 103, 39, 47, ...]`
  - Shows path through the grid

### Chain Code (Trajectory)
- **chain_code**: 26-directional movement encoding
  - Example: `[3, 4, 15, 2, 1, 8, ...]`
  - Values 0-25 represent directions in 3D space
  - Captures how hand moves through space

### Palm Angles (NEW)
- **palm_angles_left**: Left hand orientation changes
- **palm_angles_right**: Right hand orientation changes
- Format: `[[yaw, pitch, roll], [yaw, pitch, roll], ...]`
- Units: Degrees
- Recorded when change exceeds 5 degrees per axis

### Distance Tracking (NEW)
- **trigger_distance_left**: Distance from left hand to face
- **trigger_distance_right**: Distance from right hand to face
- Format: `[0.45, 0.48, 0.52, ...]`
- Units: Normalized coordinates (0-1 range)
- Recorded when change exceeds 0.05 units

---

## Tips for Better Captures

### Lighting
- Good lighting helps with hand detection
- Avoid backlighting or shadows on hands
- Face the camera directly

### Hand Position
- Keep hands fully visible in frame
- Avoid touching or crossing the midline too much
- Make clear, deliberate gestures

### Multiple Signs
- You can capture the same sign multiple times
- This builds a richer dataset for training
- Variations help with robustness

### Clearing Data
- **"Clear Session"** removes all current samples (after confirmation)
- Use this when starting a new collection batch
- Export first if you want to keep old data!

---

## Troubleshooting

### Camera Not Detected
- Check if webcam is connected
- Try another USB port
- Close other applications using the camera

### Hands Not Detected
- Improve lighting
- Keep hands fully in frame
- Avoid very fast movements initially

### Grid Not Showing
- Check that gesture capture is active
- Ensure both hands are visible
- Grid only shows hits during active capture

### Export Fails
- Check file write permissions
- Ensure disk has free space
- Try different filename

---

## Keyboard Shortcuts

While the camera preview is active:
- **ESC** - Exit the application
- **SPACE** - Toggle start/stop capture (from camera window)

---

## Dataset Best Practices

1. **Label Consistency** - Use same labels across sessions
2. **Multiple Examples** - Capture each sign 5-10 times
3. **Variation** - Capture different hand positions and speeds
4. **Regular Export** - Export data regularly to prevent loss
5. **Documentation** - Note any special conditions in labels

---

## Next Steps

- **Learn the system:** See [ARCHITECTURE.md](ARCHITECTURE.md) to understand how it works
- **Explore algorithms:** Check [CORE_CONCEPTS.md](CORE_CONCEPTS.md) for depth scaling and layer triggering
- **Develop:** Read [DEVELOPMENT.md](DEVELOPMENT.md) for code structure and APIs

---

**Version:** January 2026  
**Status:** Active
