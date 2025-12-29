# Tennis Analysis Integration System

A unified system that integrates shot detection with 3D player reconstruction for comprehensive tennis match analysis.

## Overview

This repository integrates two separate tennis analysis systems:

1. Detects shots using ball tracking, audio analysis, and homography-based court mapping  https://github.com/sssabet/tennis_analytics 
2. Performs 3D reconstruction of player movements using SAM3D  https://github.com/sssabet/tennis_3d_reconstruction

The integration system:
- Automatically detects shots in tennis videos
- Clips video segments around each detected shot
- Performs 3D reconstruction of player movements
- Provides a unified web interface for viewing results

![Uploading image.png…]()


## Repository Structure

```
tennis-analysis-integration/
├── integrate_shot_to_3d.py      # Core integration script
├── unified_viewer/              # Web-based viewer application
│   ├── app.py                   # FastAPI backend
│   ├── frontend/                # React frontend
│   └── start_backend.sh         # Backend startup script
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git ignore rules
```

## Dependencies

This repository depends on two other **separate git repositories** that should be cloned as siblings:

```
sportai/
├── tennis-analysis-integration/     # This repo (you are here)
├── tennis_shot_homography_detection/  # Shot detection system (separate repo) https://github.com/sssabet/tennis_analytics
└── tennis_single_player/             # 3D reconstruction system (separate repo)  https://github.com/sssabet/tennis_3d_reconstruction
```

**Important**: The `tennis_shot_homography_detection` and `tennis_single_player` directories are **NOT** included in this repository. They must be cloned separately as sibling directories.

## Installation

### 1. Clone This Repository

```bash
git clone <this-repo-url>
cd tennis-analysis-integration
```

### 2. Clone Dependencies

Clone the two required repositories as siblings:

```bash
cd ..
git clone https://github.com/sssabet/tennis_3d_reconstruction
git clone https://github.com/sssabet/tennis_analytics
cd tennis-analysis-integration
```

### 3. Install Python Dependencies

Install dependencies from all three repositories:

```bash
# Install shot detection dependencies
cd ../tennis_shot_homography_detection
pip install -r requirements.txt

# Install 3D reconstruction dependencies
cd ../tennis_single_player
pip install -r requirements.txt

# Install integration dependencies
cd ../tennis-analysis-integration
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd unified_viewer/frontend
npm install
```

### 5. Download Model Weights

Ensure model weights are in place:

**Shot Detection Models** (`tennis_shot_homography_detection/weights/`):
- `tracknet.pth`
- `courtside_yolo.pt`
- `court_keypoints.pth`
- `yolov8_pose.pt`

**3D Reconstruction Models** (see `tennis_single_player/MODEL_SETUP.md`):
- SAM3D Body models (automatically downloaded on first run)

## Usage

### Command Line Interface

Process a video file directly:

```bash
python integrate_shot_to_3d.py path/to/video.mp4
```

Options:
- `--output-dir`: Custom output directory (default: `integrated_outputs/`)
- `--clip-duration`: Duration of clip in seconds (default: 1.0)
- `--device`: Device for processing (`cuda`, `cpu`, or `auto`)
- `--multi-person`: Enable multi-person 3D reconstruction (default: True)
- `--generate-full-video`: Generate full analysis video with overlays (default: True)

### Web Interface

Start the unified web viewer:

```bash
# Terminal 1: Start backend
cd unified_viewer
./start_backend.sh
# Or: python app.py

# Terminal 2: Start frontend (development)
cd unified_viewer/frontend
npm start
```

Then open http://localhost:3000 in your browser.

For production, build the frontend first:

```bash
cd unified_viewer/frontend
npm run build
cd ..
python app.py
```

Then open http://localhost:8000

## Features

### Shot Detection
- Automatic shot detection using ball tracking + audio analysis
- Homography-based court mapping
- Ball trajectory visualization
- Shot timing and frame information

### 3D Reconstruction
- Full 3D player body reconstruction for each shot
- Frame-by-frame playback of movements
- Velocity and acceleration analysis
- Multi-person support

### Web Interface
- Upload tennis videos
- View all detected shots in a grid
- Interactive 3D viewer with WASD controls
- Real-time processing status
- Full analysis video playback

## Credits

- Shot detection: TrackNet, CourtSide, YOLOv8
- 3D reconstruction: MediaPipe, SAM3D Body
- Web framework: FastAPI, React, Three.js

