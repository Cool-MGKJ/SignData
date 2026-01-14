# SignData Documentation Index

Welcome to the SignData project documentation. This folder contains comprehensive guides for understanding, using, and developing the ASL gesture recognition system.

## Quick Navigation

### 📖 Getting Started
- **[USER_GUIDE.md](USER_GUIDE.md)** - How to install, run, and use the application
  - Installation steps
  - Using the UI
  - Capturing gestures
  - Exporting datasets

### 🏗️ System Design
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture and design
  - 2-layer voxel grid system
  - Face-centered coordinate space
  - Head tracking and rotation
  - Grid visualization
  - Component interactions

- **[CORE_CONCEPTS.md](CORE_CONCEPTS.md)** - Key algorithms and concepts
  - Dynamic depth scaling algorithm
  - Layer triggering logic
  - 26-directional chain code system
  - Hand distance tracking
  - Palm angle tracking

### 💻 Development
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Code structure and APIs
  - Module overview
  - Public API reference
  - Class hierarchies
  - Method signatures
  - Error handling

- **[CODE_CHANGES.md](CODE_CHANGES.md)** - Recent implementation changes
  - Palm angle tracking implementation
  - Distance tracking integration
  - Code change reference

### 📊 Project History
- **[PROJECT_HISTORY.md](PROJECT_HISTORY.md)** - Evolution and major phases
  - Development phases (11 total)
  - Key technical decisions
  - Version history
  - Known issues and resolutions
  - Future roadmap

---

## 📚 Complete Documentation Summary

| Document | Type | Focus | Pages |
|----------|------|-------|-------|
| **USER_GUIDE.md** | Tutorial | Installation, usage, tips | 10 |
| **ARCHITECTURE.md** | Design | System components, layers, pipeline | 14 |
| **CORE_CONCEPTS.md** | Reference | Algorithms, formulas, calculations | 16 |
| **DEVELOPMENT.md** | Technical | Code structure, APIs, examples | 12 |
| **CODE_CHANGES.md** | Changelog | Recent implementations with diffs | 10 |
| **PROJECT_HISTORY.md** | Timeline | 11 phases, decisions, metrics | 12 |

**Total Coverage:** ~3,700 lines, 7 complete documents

---

## System Overview

**SignData** is an ASL dataset collection tool that:
1. Tracks hand landmarks using MediaPipe
2. Maps hands to a 3D face-centered voxel grid (160 voxels in 2 layers)
3. Records spatial hits, movement trajectory (chain code), hand orientation (angles), and distance metrics
4. Exports labeled samples for machine learning

**Key Statistics:**
- Grid: 8 wide × 10 tall × 2 deep = 160 voxels total
- Layers: Layer 0 (face) + Layer 1 (camera)
- Tracking: 21 hand landmarks per hand, face mesh reference
- Output: Normalized coordinates + spatial hits + chain codes + angles + distances

---

## Common Tasks

### Learn the System Architecture
→ Start with [ARCHITECTURE.md](ARCHITECTURE.md)

### Understand How Gestures Are Tracked
→ Read [CORE_CONCEPTS.md](CORE_CONCEPTS.md) - "Layer Triggering Logic"

### Run the Application
→ Follow [USER_GUIDE.md](USER_GUIDE.md) - "Getting Started"

### Modify or Extend the Code
→ Review [DEVELOPMENT.md](DEVELOPMENT.md) for APIs and structure

### Track Implementation Changes
→ Check [CODE_CHANGES.md](CODE_CHANGES.md) and [PROJECT_HISTORY.md](PROJECT_HISTORY.md)

---

## Document Structure

Each document is self-contained but cross-references others:

```
USER_GUIDE.md ─────→ "See ARCHITECTURE for how it works"
    ↓
ARCHITECTURE.md ────→ "Layer system explained in CORE_CONCEPTS"
    ↓
CORE_CONCEPTS.md ───→ "Implementation details in DEVELOPMENT"
    ↓
DEVELOPMENT.md ─────→ "API signatures and code structure"
    ↓
CODE_CHANGES.md ────→ "Recent feature additions"
    ↓
PROJECT_HISTORY.md ─→ "Why decisions were made"
```

---

## File Organization

```
docs/
├── index.md                    # This file
├── USER_GUIDE.md              # How to use the app
├── ARCHITECTURE.md             # System design
├── CORE_CONCEPTS.md           # Algorithms & concepts
├── DEVELOPMENT.md             # Code structure & APIs
├── CODE_CHANGES.md            # Recent changes
└── PROJECT_HISTORY.md         # Evolution & phases
```

---

## Key Concepts at a Glance

### The Voxel Grid
A 3D grid (8×10×2 = 160 voxels) centered on the face that tracks which points are touched by hands.

### Dynamic Depth Scaling
Automatically determines which layer (face or camera) to use based on hand distance:
```
relative_z = (hand_z - face_z_reference) / reference_length
```
- `relative_z >= 0` → Layer 0 (GREEN, inner layer near face)
- `relative_z < 0` → Layer 1 (RED, outer layer near camera)

### Chain Code
A sequence of 26-directional movement vectors recording how the hand moves through space.

### Multi-Dimensional Tracking
Each gesture now records:
1. **Spatial** - Which voxels were hit
2. **Trajectory** - 26-direction chain code
3. **Orientation** - Palm yaw/pitch/roll angles (NEW)
4. **Distance** - Hand distance from face (NEW)

---

## Quick Links to Key Content

| Topic | Location |
|-------|----------|
| Installation | USER_GUIDE.md → Installation |
| Running the app | USER_GUIDE.md → Getting Started |
| Layer system | ARCHITECTURE.md → 2-Layer System |
| Grid visualization | ARCHITECTURE.md → Visualization |
| Depth scaling math | CORE_CONCEPTS.md → Dynamic Depth Scaling |
| Layer triggering | CORE_CONCEPTS.md → Layer Triggering Logic |
| Hand landmarks | DEVELOPMENT.md → Hand Tracking |
| API methods | DEVELOPMENT.md → Public API |
| Recent changes | CODE_CHANGES.md |
| Version history | PROJECT_HISTORY.md |

---

## Getting Help

1. **How do I...?** → Check the relevant guide in USER_GUIDE.md
2. **How does...work?** → Look in CORE_CONCEPTS.md for algorithms
3. **Where is the code for...?** → Find it in DEVELOPMENT.md
4. **What changed recently?** → See CODE_CHANGES.md and PROJECT_HISTORY.md

---

**Last Updated:** January 14, 2026  
**Status:** Active Development
