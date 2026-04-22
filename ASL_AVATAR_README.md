# ASL Avatar Dataset Creation Tool

A standalone tool for capturing ASL gesture motion data for 3D avatar animation. This tool is completely separate from the SignData project and captures raw, structured motion data without normalization or classification.

## Features

- **Upper Body Pose Capture**: Captures 10 key joints (pelvis, spine, neck, head, shoulders, elbows, wrists)
- **Hand Tracking**: Captures 21 landmarks per hand (left and right separately)
- **Facial Expressions**: Captures facial blendshape weights (smile, mouthOpen, browRaise, eyeBlink, etc.)
- **Frame-by-Frame Sequences**: Stores time-indexed gesture sequences
- **Label-Driven**: One gesture per capture with user-defined labels
- **JSON Export**: Exports data in avatar-friendly JSON format

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe>=0.10.14`)
- NumPy (`numpy>=1.24.0`)
- Pillow (`Pillow>=10.0.0`)
- Tkinter (usually included with Python)

## Installation

```bash
# Install dependencies
pip install opencv-python mediapipe numpy Pillow
```

## Usage

### Running the Application

```bash
python asl_avatar_main.py
```

### Capture Workflow

1. **Start Capture**: Click "Start Capture" button
2. **Perform Gesture**: Perform your ASL sign in front of the camera
3. **Stop Capture**: Click "Stop Capture" when finished
4. **Enter Label**: Type the gesture label (lowercase, snake_case, e.g., "hello", "thank_you")
5. **Save Gesture**: Click "Save Gesture" to export to JSON

### Output Format

Each gesture is saved as a separate JSON file in the output directory:

```
asl_avatar_dataset/
├── hello.json
├── thank_you.json
├── please.json
└── ...
```

### JSON Structure

```json
{
  "label": "hello",
  "fps": 30,
  "num_frames": 42,
  "frames": [
    {
      "timestamp": 0.000,
      "body_pose": {
        "joints": [[x, y, z], ...],
        "hierarchy": ["pelvis", "spine", "neck", "head", ...]
      },
      "left_hand": [[x, y, z], ...],
      "right_hand": [[x, y, z], ...],
      "face": {
        "blendshapes": {
          "smile": 0.6,
          "mouthOpen": 0.2,
          "browRaise": 0.1,
          "eyeBlinkLeft": 0.0,
          "eyeBlinkRight": 0.0,
          "jawOpen": 0.0
        }
      }
    }
  ]
}
```

## Body Pose Joints

The tool captures 10 upper body joints in hierarchical order (legs and lower body excluded):

1. pelvis (torso base)
2. spine
3. neck
4. head
5. left_shoulder
6. left_elbow
7. left_wrist
8. right_shoulder
9. right_elbow
10. right_wrist

## Hand Landmarks

Each hand is captured with 21 landmarks (MediaPipe standard):
- 0-4: Thumb
- 5-8: Index finger
- 9-12: Middle finger
- 13-16: Ring finger
- 17-20: Pinky

## Facial Blendshapes

The tool captures the following facial expression blendshapes (normalized 0.0-1.0):

- `smile`: Overall smile intensity
- `mouthOpen`: Mouth opening amount
- `browRaise`: Eyebrow raise
- `eyeBlinkLeft`: Left eye blink
- `eyeBlinkRight`: Right eye blink
- `jawOpen`: Jaw opening
- `mouthSmileLeft`: Left side smile
- `mouthSmileRight`: Right side smile
- `browInnerUp`: Inner brow raise
- `eyeSquintLeft`: Left eye squint
- `eyeSquintRight`: Right eye squint

## Labeling Rules

- Labels must be **lowercase**
- Use **snake_case** (underscores, not spaces or hyphens)
- Examples: `hello`, `thank_you`, `please`, `i_love_you`
- Invalid: `Hello`, `thank-you`, `thank you`

## Important Notes

- This tool does **NOT** perform gesture classification
- This tool does **NOT** normalize to face-centered space
- This tool does **NOT** use voxel grids
- This tool captures **raw motion data** for avatar animation
- One gesture per capture session
- Unknown or invalid captures should be discarded

## Troubleshooting

### Camera Not Working
- Check camera permissions
- Try changing `camera_index` in `asl_avatar_main.py` (default: 0)

### No Hand Detection
- Ensure hands are clearly visible
- Improve lighting
- Check camera focus

### Poor Face Detection
- Face should be clearly visible
- Ensure good lighting
- Remove obstructions (glasses, masks, etc.)

## Architecture

```
asl_avatar_main.py      # Main UI application
asl_avatar_capture.py   # Core capture engine
asl_avatar_export.py     # JSON export module
```

## License

This tool is standalone and independent from the SignData project.
