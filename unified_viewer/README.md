# Unified Tennis Analysis Viewer

A complete web-based system that integrates shot detection with 3D player reconstruction.

> **Note**: This is part of the tennis-analysis-integration repository. See the main [README.md](../README.md) for full installation and setup instructions.

## Features

### 🎯 Shot Detection
- Automatic shot detection using ball tracking + audio analysis
- Homography-based court mapping
- Ball trajectory visualization in court coordinates
- Shot timing and frame information

### 🎨 3D Reconstruction
- Full 3D player body reconstruction for each shot
- Frame-by-frame playback of movements
- Velocity and acceleration analysis
- Interactive 3D viewer with rotation, zoom, and pan

### 🌐 Unified Web Interface
- Upload tennis videos
- View all detected shots in a grid
- Click any shot to view its 3D reconstruction
- Real-time processing status
- Responsive design

## Installation

### Backend

1. **Install Python dependencies** (from both repos):
```bash
# Install shot detection dependencies
cd ../tennis_shot_homography_detection
pip install -r requirements.txt

# Install 3D reconstruction dependencies
cd ../tennis_single_player
pip install -r requirements.txt
```

2. **Install additional unified viewer dependencies**:
```bash
pip install fastapi uvicorn python-multipart
```

3. **Ensure model weights are in place**:
- `tennis_shot_homography_detection/weights/` should contain:
  - `tracknet.pth`
  - `courtside_yolo.pt`
  - `court_keypoints.pth`
  - `yolov8_pose.pt`

### Frontend

1. **Install Node.js dependencies**:
```bash
cd frontend
npm install
```

2. **Build frontend** (for production):
```bash
npm run build
```

## Usage

### Start the Unified Viewer

#### Option 1: Development Mode (Recommended for testing)

Start backend (in one terminal):
```bash
cd unified_viewer
python app.py
```

Start frontend dev server (in another terminal):
```bash
cd unified_viewer/frontend
npm start
```

Then open: http://localhost:3000

#### Option 2: Production Mode

Build frontend and run backend only:
```bash
# Build frontend
cd unified_viewer/frontend
npm run build

# Start backend (serves frontend automatically)
cd ..
python app.py
```

Then open: http://localhost:8000

### Using the Web Interface

1. **Upload Video**
   - Click "Choose Video File" and select a tennis match video
   - Adjust clip duration (default: 1.0 seconds per shot)
   - Enable/disable multi-person mode
   - Click "Upload & Analyze"

2. **Wait for Processing**
   - System detects shots using homography and audio
   - Each shot is clipped and sent for 3D reconstruction
   - Progress is shown in real-time

3. **View Detected Shots**
   - All shots displayed in a grid with thumbnails
   - Each card shows shot number, time, frame, and velocity info
   - Green badge indicates 3D reconstruction is available

4. **View 3D Reconstruction**
   - Click any shot card to view its 3D reconstruction
   - Use mouse to rotate, zoom, and pan the 3D view
   - Slider to navigate through frames
   - Sidebar shows swing analysis and velocity profile

### Controls in 3D Viewer

- **Left Click + Drag**: Rotate view
- **Right Click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out
- **Slider**: Change frame

## API Endpoints

### Upload & Analysis
- `POST /api/upload` - Upload video
- `POST /api/analyze/{job_id}` - Start analysis
- `GET /api/status/{job_id}` - Check progress
- `GET /api/results/{job_id}` - Get complete results

### Shots
- `GET /api/shots/{job_id}` - List all detected shots
- `GET /api/shots/{job_id}/{shot_number}/thumbnail` - Get shot thumbnail
- `GET /api/shots/{job_id}/{shot_number}/3d` - Get 3D reconstruction data
- `GET /api/shots/{job_id}/{shot_number}/models/{filename}` - Download 3D model
- `GET /api/shots/{job_id}/{shot_number}/velocity_plot` - Get velocity plot

## Output Structure

```
unified_outputs/
├── {job_id}/
│   ├── clips/
│   │   ├── shot_001_clip.mp4         # Video clip for shot 1
│   │   ├── shot_001_clip.json        # Homography metadata
│   │   └── ...
│   ├── shot_001/
│   │   ├── meshes_3d/                # 3D reconstructions (GLB files)
│   │   ├── swing_frames/             # Extracted frames
│   │   ├── velocity_profile.png      # Velocity plot
│   │   └── analysis_report.json      # Full analysis
│   └── processing_summary.json       # Overall summary
└── uploads/                          # Uploaded videos
```

## Architecture

### Backend (FastAPI)
- Integrates both shot detection and 3D reconstruction systems
- Handles video uploads and storage
- Manages background processing tasks
- Serves API endpoints for frontend
- Serves static frontend files in production

### Frontend (React)
- Upload interface with settings
- Real-time progress tracking
- Responsive shots grid with thumbnails
- Interactive 3D viewer using Three.js
- React Three Fiber for 3D rendering

### Integration
- Uses `integrate_shot_to_3d.py` for core processing
- Shot detection from `tennis_shot_homography_detection`
- 3D reconstruction from `tennis_single_player`
- Homography data preserved and passed between systems

## Configuration

Edit settings in the upload form:
- **Clip Duration**: 0.5 - 3.0 seconds (default: 1.0)
- **Multi-Person Mode**: Enable to detect multiple players
- **Device**: Automatically selects GPU if available

## Troubleshooting

**Backend won't start:**
- Ensure both repos' dependencies are installed
- Check that model weights are in place
- Verify Python 3.8+ is installed

**Frontend won't build:**
- Delete `node_modules` and run `npm install` again
- Check Node.js version (14+ required)

**No shots detected:**
- Check video has clear ball visibility
- Verify audio track exists
- Ensure court is visible in video

**3D reconstruction fails:**
- Check player is clearly visible
- Try increasing clip duration
- Verify SAM3D models are installed (see tennis_single_player/MODEL_SETUP.md)

**Slow processing:**
- Use GPU for faster processing (CUDA)
- Reduce clip duration
- Process shorter videos

## Development

### Frontend Development
```bash
cd frontend
npm start  # Hot reload enabled
```

### Backend Development
```bash
python app.py  # Auto-reloads with uvicorn --reload
```

### Adding New Features
- Backend routes: `unified_viewer/app.py`
- Frontend components: `unified_viewer/frontend/src/components/`
- Styles: Corresponding `.css` files

## License

This system integrates multiple open-source components:
- Tennis shot detection system
- Tennis 3D reconstruction system
- React, Three.js, FastAPI (MIT/Apache licenses)

## Credits

- Shot detection: TrackNet, CourtSide, YOLOv8
- 3D reconstruction: MediaPipe, SAM3D Body
- Web framework: FastAPI, React, Three.js


