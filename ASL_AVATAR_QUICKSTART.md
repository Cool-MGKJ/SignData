# ASL Avatar Dataset Tool - Quick Start

## Run the Tool

```bash
python asl_avatar_main.py
```

## Basic Workflow

1. **Start Capture** → Click button
2. **Perform Gesture** → Sign in front of camera
3. **Stop Capture** → Click button  
4. **Enter Label** → Type label (e.g., `hello`, `thank_you`)
5. **Save Gesture** → Exports to JSON

## Output Location

Default: `asl_avatar_dataset/` directory

Files: `{label}.json` (e.g., `hello.json`, `thank_you.json`)

## Label Format

- ✅ `hello`, `thank_you`, `i_love_you`
- ❌ `Hello`, `thank-you`, `thank you`

## Data Captured Per Frame

- **Body**: 10 upper body joints (pelvis → head → arms only, no legs)
- **Left Hand**: 21 landmarks (or zeros if not detected)
- **Right Hand**: 21 landmarks (or zeros if not detected)  
- **Face**: 11 blendshapes (smile, mouthOpen, browRaise, etc.)

## Example JSON Output

```json
{
  "label": "hello",
  "fps": 30,
  "num_frames": 42,
  "frames": [
    {
      "timestamp": 0.000,
      "body_pose": {
        "joints": [[0.5, 0.6, 0.1], ...],
        "hierarchy": ["pelvis", "spine", ...]
      },
      "left_hand": [[0.3, 0.4, 0.2], ...],
      "right_hand": [[0.7, 0.4, 0.2], ...],
      "face": {
        "blendshapes": {
          "smile": 0.6,
          "mouthOpen": 0.2
        }
      }
    }
  ]
}
```

## Troubleshooting

**Camera not working?**
- Check camera permissions
- Try different camera index in code

**No hand detection?**
- Ensure hands are clearly visible
- Improve lighting

**Poor face detection?**
- Face should be clearly visible
- Good lighting required
