# ASL Avatar Playback - Quick Start

## Installation

```bash
pip install -r requirements_playback.txt
```

## Run

```bash
python asl_avatar_playback_main.py
```

## Basic Workflow

1. **Start the application**
   ```bash
   python asl_avatar_playback_main.py
   ```

2. **Enter text to play**
   - Press **T** key
   - Type gesture text (e.g., "hello sorry")
   - Press Enter

3. **Play gestures**
   - Press **SPACE** to start playback
   - Press **SPACE** again to stop

4. **Exit**
   - Press **ESC** to exit

## Command Line Usage

```bash
# Play text immediately
python asl_avatar_playback_main.py --text "hello sorry"

# Custom dataset directory
python asl_avatar_playback_main.py --dataset /path/to/dataset --text "thank you"
```

## Example

```bash
# 1. Create some gestures with the capture tool
# (creates hello.json, sorry.json, etc.)

# 2. Play them back
python asl_avatar_playback_main.py --text "hello sorry"
```

## Controls

| Key | Action |
|-----|--------|
| **T** | Enter text input |
| **SPACE** | Start/Stop playback |
| **R** | Reset |
| **ESC** | Exit |

## Troubleshooting

**No window appears?**
- Check OpenGL/GLFW installation
- Try: `pip install --upgrade PyOpenGL glfw`

**Gestures not found?**
- Check dataset directory path
- Ensure JSON files exist
- Verify gesture labels match text

**Rendering issues?**
- Update graphics drivers
- Check OpenGL version: `glxinfo | grep "OpenGL version"` (Linux)
