# ASL Avatar System - Complete Overview

This document provides an overview of the complete ASL Avatar system, including both the dataset creation tool and the playback system.

 python asl_avatar_playback_main.py --dataset "c:\fy-project\SignData\asl_avatar_dataset" --text "test2"

python asl_avatar_main.py         

## System Components

### 1. Dataset Creation Tool

**Purpose**: Capture ASL gesture motion data for avatar animation.

**Files**:
- `asl_avatar_capture.py` - Core capture engine
- `asl_avatar_export.py` - JSON export module
- `asl_avatar_main.py` - Main UI application

**Features**:
- Captures upper body pose (10 joints)
- Captures hand landmarks (21 per hand)
- Captures facial expressions (11 blendshapes)
- Exports to JSON format

**Usage**:
```bash
python asl_avatar_main.py
```

### 2. Playback System

**Purpose**: Reconstruct and animate 3D avatar from gesture dataset.

**Files**:
- `asl_avatar_loader.py` - JSON file loader
- `asl_avatar_text_parser.py` - Text to gesture mapping
- `asl_avatar_renderer.py` - 3D rendering engine
- `asl_avatar_player.py` - Playback controller
- `asl_avatar_playback_main.py` - Main playback application

**Features**:
- Text input parsing
- Sequential gesture playback
- 3D avatar rendering
- Temporal fidelity preservation

**Usage**:
```bash
python asl_avatar_playback_main.py --text "hello sorry"
```

## Complete Workflow

### Step 1: Create Dataset

1. Run capture tool: `python asl_avatar_main.py`
2. Start capture
3. Perform ASL gesture
4. Stop capture
5. Enter label (e.g., "hello")
6. Save gesture

Repeat for all desired gestures.

### Step 2: Playback Gestures

1. Run playback system: `python asl_avatar_playback_main.py`
2. Enter text: Press **T**, type "hello sorry"
3. Play: Press **SPACE**

## Data Format

Each gesture is stored as a JSON file:

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
      "left_hand": [[x, y, z], ...],  // 21 landmarks
      "right_hand": [[x, y, z], ...], // 21 landmarks
      "face": {
        "blendshapes": {
          "smile": 0.6,
          "mouthOpen": 0.2,
          ...
        }
      }
    }
  ]
}
```

## Architecture

```
┌─────────────────────────────────────┐
│     Dataset Creation Tool            │
│  ┌──────────┐  ┌──────────┐        │
│  │ Capture  │→ │  Export  │        │
│  └──────────┘  └──────────┘        │
│         ↓                            │
│    JSON Files                        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      Playback System                 │
│  ┌──────────┐  ┌──────────┐        │
│  │  Loader  │→ │  Parser  │        │
│  └──────────┘  └──────────┘        │
│         ↓                            │
│  ┌──────────┐  ┌──────────┐     │
│  │  Player   │→ │ Renderer  │     │
│  └──────────┘  └──────────┘     │
│         ↓                            │
│    3D Avatar                         │
└─────────────────────────────────────┘
```

## Requirements

### Capture Tool
```bash
pip install opencv-python mediapipe numpy Pillow
```

### Playback System
```bash
pip install PyOpenGL glfw numpy
```

## Key Features

### Motion Fidelity
- **No ML inference**: Direct reconstruction from recorded data
- **Frame-accurate timing**: Preserves exact temporal information
- **No compression**: Raw motion data maintained

### Data Modalities
- **Body Pose**: 10 upper body joints
- **Hands**: 21 landmarks per hand
- **Face**: 11 blendshape values

### Playback Features
- Sequential gesture playback
- Text-to-gesture mapping
- 3D visualization
- Real-time rendering

## File Structure

```
SignData/
├── asl_avatar_capture.py          # Capture engine
├── asl_avatar_export.py           # JSON export
├── asl_avatar_main.py              # Capture UI
├── asl_avatar_loader.py            # JSON loader
├── asl_avatar_text_parser.py      # Text parser
├── asl_avatar_renderer.py          # 3D renderer
├── asl_avatar_player.py            # Playback controller
├── asl_avatar_playback_main.py     # Playback UI
├── asl_avatar_dataset/             # Dataset directory
│   ├── hello.json
│   ├── sorry.json
│   └── ...
├── requirements.txt                # Capture dependencies
└── requirements_playback.txt       # Playback dependencies
```

## Documentation

- `ASL_AVATAR_README.md` - Capture tool documentation
- `ASL_AVATAR_QUICKSTART.md` - Capture quick start
- `ASL_AVATAR_PLAYBACK_README.md` - Playback documentation
- `ASL_AVATAR_PLAYBACK_QUICKSTART.md` - Playback quick start

## Example Usage

### Creating a Gesture

```bash
# 1. Start capture tool
python asl_avatar_main.py

# 2. In UI:
#    - Click "Start Capture"
#    - Perform "hello" gesture
#    - Click "Stop Capture"
#    - Enter label: "hello"
#    - Click "Save Gesture"
```

### Playing Gestures

```bash
# Command line
python asl_avatar_playback_main.py --text "hello sorry"

# Or interactive
python asl_avatar_playback_main.py
# Press T, enter "hello sorry"
# Press SPACE to play
```

## Notes

- The system is **completely separate** from the SignData project
- No dependencies on voxel grids or normalization
- Pure reconstruction system (no ML)
- Designed for avatar animation pipelines
