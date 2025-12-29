# Setup Guide

This guide will help you set up the tennis analysis integration system.

## Prerequisites

- Python 3.8 or higher
- Node.js 14+ and npm
- CUDA-capable GPU (optional, but recommended for faster processing)
- Git

## Step-by-Step Setup

### 1. Clone All Repositories

Clone all three repositories as siblings:

```bash
mkdir -p ~/sportai
cd ~/sportai

# Clone this integration repository
git clone <tennis-analysis-integration-url>
cd tennis-analysis-integration

# Clone dependencies (as siblings)
cd ..
git clone <tennis_shot_homography_detection-url>
git clone <tennis_single_player-url>
cd tennis-analysis-integration
```

Your directory structure should look like:

```
sportai/
├── tennis-analysis-integration/     # This repo
├── tennis_shot_homography_detection/  # Shot detection
└── tennis_single_player/             # 3D reconstruction
```

### 2. Install Python Dependencies

Install dependencies from all three repositories:

```bash
# Shot detection dependencies
cd ../tennis_shot_homography_detection
pip install -r requirements.txt

# 3D reconstruction dependencies
cd ../tennis_single_player
pip install -r requirements.txt

# Integration dependencies
cd ../tennis-analysis-integration
pip install -r requirements.txt
```

**Note**: If using conda, you may need to install dependencies in the conda environment:

```bash
conda activate ml-backend  # or your environment name
# Then install dependencies as above
```

### 3. Install Frontend Dependencies

```bash
cd unified_viewer/frontend
npm install
cd ../..
```

### 4. Download Model Weights

#### Shot Detection Models

Place the following files in `tennis_shot_homography_detection/weights/`:

- `tracknet.pth` - TrackNet model weights
- `courtside_yolo.pt` - CourtSide YOLO model
- `court_keypoints.pth` - Court keypoint detection model
- `yolov8_pose.pt` - YOLOv8 pose estimation model

#### 3D Reconstruction Models

Follow the instructions in `tennis_single_player/MODEL_SETUP.md` to download SAM3D models. They will be automatically downloaded on first run.

### 5. Verify Installation

Test the integration script:

```bash
python integrate_shot_to_3d.py --help
```

You should see the help message with available options.

## Running the System

### Command Line Usage

Process a video:

```bash
python integrate_shot_to_3d.py path/to/video.mp4
```

### Web Interface

Start the backend:

```bash
cd unified_viewer
./start_backend.sh
```

In another terminal, start the frontend:

```bash
cd unified_viewer
./start_frontend.sh
```

Then open http://localhost:3000 in your browser.

## Troubleshooting

### Import Errors

If you get import errors like "No module named 'TennisAnalyzer'":

1. Verify both dependency repos are cloned as siblings
2. Check that the directory structure matches the expected layout
3. Ensure all dependencies are installed

### CUDA/GPU Issues

If you want to use CPU instead of GPU:

```bash
python integrate_shot_to_3d.py video.mp4 --device cpu
```

### Frontend Build Errors

If the frontend won't build:

```bash
cd unified_viewer/frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Model Not Found Errors

- Verify model weights are in the correct locations
- Check file permissions
- Ensure models are downloaded completely

## Next Steps

- Read the main [README.md](README.md) for detailed usage instructions
- Check the [unified_viewer/README.md](unified_viewer/README.md) for web interface details
- Try processing a sample video to verify everything works

