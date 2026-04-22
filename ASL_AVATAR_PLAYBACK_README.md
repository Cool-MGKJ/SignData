# ASL Avatar Playback System

A 3D avatar reconstruction and animation system that plays back ASL gestures from the dataset.

## Overview

This system reconstructs and animates a 3D human avatar using gesture data captured by the ASL Avatar Dataset Creation Tool. It accepts text input, maps words to gesture files, and plays back the gestures sequentially with correct timing.

## Features

- **Text-to-Gesture Mapping**: Maps text input to gesture sequences
- **Sequential Playback**: Plays multiple gestures in sequence
- **3D Avatar Rendering**: Renders body pose, hands, and facial expressions
- **Temporal Fidelity**: Preserves exact motion timing from dataset
- **No ML Inference**: Direct reconstruction from recorded data

## Requirements

```bash
pip install PyOpenGL glfw numpy
```

### System Requirements

- Python 3.8+
- OpenGL support (usually available on most systems)
- GLFW for windowing

## Usage

### Basic Usage

```bash
# Run the playback system
python asl_avatar_playback_main.py

# With text input
python asl_avatar_playback_main.py --text "hello sorry"

# Custom dataset directory
python asl_avatar_playback_main.py --dataset /path/to/dataset
```

### Interactive Controls

- **T** - Enter text to play
- **SPACE** - Start/Stop playback
- **R** - Reset
- **ESC** - Exit

### Programmatic Usage

```python
from asl_avatar_loader import GestureLoader
from asl_avatar_text_parser import TextParser
from asl_avatar_player import GesturePlayer
from asl_avatar_renderer import AvatarRenderer

# Initialize components
loader = GestureLoader("asl_avatar_dataset")
parser = TextParser(loader)
renderer = AvatarRenderer()
player = GesturePlayer(renderer, parser)

# Load and play text
player.load_text("hello sorry")
player.start_playback()

# Update loop
while player.is_playing:
    player.update()
    # Render frame (handled by player)
```

## Architecture

### Components

1. **GestureLoader** (`asl_avatar_loader.py`)
   - Loads JSON gesture files
   - Caches loaded gestures
   - Validates data structure

2. **TextParser** (`asl_avatar_text_parser.py`)
   - Parses text input
   - Maps words to gesture files
   - Handles multi-word gestures

3. **AvatarRenderer** (`asl_avatar_renderer.py`)
   - Renders 3D body skeleton
   - Renders hand landmarks
   - Visualizes facial expressions

4. **GesturePlayer** (`asl_avatar_player.py`)
   - Manages playback state
   - Handles timing and sequencing
   - Coordinates rendering

5. **PlaybackApp** (`asl_avatar_playback_main.py`)
   - Main application window
   - User interface
   - Event handling

## Data Flow

```
Text Input
    ↓
TextParser → Maps to gesture labels
    ↓
GestureLoader → Loads JSON files
    ↓
GesturePlayer → Sequences and times gestures
    ↓
AvatarRenderer → Renders 3D avatar
    ↓
Display
```

## Gesture File Format

The system expects JSON files in the format created by the capture tool:

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
        "hierarchy": ["pelvis", "spine", ...]
      },
      "left_hand": [[x, y, z], ...],
      "right_hand": [[x, y, z], ...],
      "face": {
        "blendshapes": {...}
      }
    }
  ]
}
```

## Rendering Details

### Body Pose
- 10 upper body joints (pelvis, spine, neck, head, shoulders, elbows, wrists)
- Skeleton visualization with joints and bones
- Hierarchical structure preserved

### Hands
- 21 landmarks per hand
- Full finger articulation
- Connected skeleton visualization

### Facial Expressions
- 11 blendshape values
- Visualized as colored indicators
- Normalized 0.0-1.0 range

## Limitations

- Requires OpenGL/GLFW for 3D rendering
- Facial expression visualization is simplified (not full mesh)
- No interpolation between frames (frame-by-frame playback)
- Text parsing is basic (word-level matching)

## Troubleshooting

### OpenGL Not Available
- Install PyOpenGL: `pip install PyOpenGL`
- Check graphics drivers
- Try software rendering fallback

### GLFW Not Available
- Install GLFW: `pip install glfw`
- On Linux, may need: `sudo apt-get install libglfw3`

### No Gestures Found
- Check dataset directory path
- Ensure JSON files are in correct format
- Verify gesture labels match text input

## Future Enhancements

- Frame interpolation for smoother animation
- Full facial mesh rendering
- Better text parsing (phrase recognition)
- Gesture blending between sequences
- Export to animation formats (FBX, BVH)
