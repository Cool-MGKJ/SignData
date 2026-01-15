# SignData: ASL Gesture Recognition Dataset Collection Tool

A computer vision-based system for collecting American Sign Language (ASL) gesture data using MediaPipe hand/pose detection and a 3D voxel grid tracking system.

##  Documentation

**All documentation has been consolidated into the /docs/ folder:**

### ** [Start Here: Documentation Index](docs/index.md)**

---

## Quick Links

- **[Installation & Usage Guide](docs/USER_GUIDE.md)** - How to install and run
- **[System Architecture](docs/ARCHITECTURE.md)** - How the system works
- **[Algorithm Reference](docs/CORE_CONCEPTS.md)** - Technical algorithms and formulas  
- **[Developer Guide](docs/DEVELOPMENT.md)** - Code structure and APIs
- **[Recent Changes](docs/CODE_CHANGES.md)** - New feature implementations
- **[Project Timeline](docs/PROJECT_HISTORY.md)** - Development history and phases

---

## Quick Start

`ash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python main.py

# 3. Capture gestures
# - Click Start Capture 
# - Perform your ASL gesture
# - Click Stop Capture
# - Enter a label and save

# 4. Export your data
# - Click Export Dataset
# - Choose JSON or CSV format
`

For detailed instructions, see [USER_GUIDE.md](docs/USER_GUIDE.md)

---

## Features

-  Real-time hand tracking with MediaPipe
-  Face-centered 3D voxel grid system (160 voxels, 2 layers)
-  Palm angle tracking (yaw, pitch, roll)
-  Hand-to-face distance tracking
-  Gesture trajectory recording (chain codes)
-  Multiple export formats (JSON, CSV)
-  User-friendly desktop UI

---

## System Overview

| Component | Detail |
|-----------|--------|
| Grid Size | 8  10  2 (160 voxels) |
| Layers | Layer 0 (face) + Layer 1 (camera) |
| Hand Tracking | 21 landmarks per hand (MediaPipe) |
| Head Reference | Face mesh (68+ landmarks) |
| Output Data | Coordinates, hits, trajectory, angles, distances |

---

## Archive Note

Old documentation files have been consolidated into the /docs/ folder. See [Documentation Index](docs/index.md) for the mapping.

---

**For questions or issues, refer to [Documentation Index](docs/index.md) or relevant guide above.**
