"""
Unified FastAPI Backend for Tennis Shot Detection + 3D Reconstruction
Combines both systems into a single web interface
"""
import sys
from pathlib import Path

# Add both repos to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "tennis_shot_homography_detection"))
sys.path.insert(0, str(parent_dir / "tennis_single_player"))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
import shutil
import uuid
import logging
import json
import cv2
import numpy as np
import asyncio
from datetime import datetime

# Import integration script
sys.path.insert(0, str(parent_dir))
from integrate_shot_to_3d import ShotTo3DIntegrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Unified Tennis Analysis API",
    description="Shot detection with homography + 3D reconstruction viewer",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
OUTPUT_DIR = parent_dir / "unified_outputs"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Unified Tennis Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "analyze": "/api/analyze/{job_id}",
            "status": "/api/status/{job_id}",
            "results": "/api/results/{job_id}",
            "shots": "/api/shots/{job_id}",
            "shot_details": "/api/shots/{job_id}/{shot_number}",
            "shot_3d": "/api/shots/{job_id}/{shot_number}/3d"
        }
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a tennis video for shot detection and 3D reconstruction
    
    Returns:
        job_id for tracking the analysis
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: MP4, AVI, MOV, MKV"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
    
    try:
        with open(upload_path, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    # Create job
    JOBS[job_id] = {
        'job_id': job_id,
        'filename': file.filename,
        'upload_path': str(upload_path),
        'status': 'uploaded',
        'created_at': datetime.now().isoformat(),
        'output_dir': str(OUTPUT_DIR / job_id)
    }
    
    logger.info(f"Video uploaded: {job_id} - {file.filename}")
    
    return {
        'job_id': job_id,
        'filename': file.filename,
        'status': 'uploaded'
    }


def run_analysis_task(job_id: str, clip_duration: float, device: str, multi_person: bool):
    """Background task to run shot detection and 3D reconstruction"""
    try:
        job = JOBS[job_id]
        job['status'] = 'processing'
        job['progress'] = 'Detecting shots...'
        
        # Initialize integrator
        integrator = ShotTo3DIntegrator(
            video_path=job['upload_path'],
            output_base_dir=Path(job['output_dir'])
        )
        
        # Run full pipeline
        job['progress'] = 'Running shot detection and 3D reconstruction...'
        summary = integrator.process_all_shots(
            clip_duration_s=clip_duration,
            device=device,
            multi_person=multi_person,
            generate_full_video=True  # Generate full analysis video with overlays
        )
        
        # Update job status
        job['status'] = 'complete'
        job['results'] = summary
        job['completed_at'] = datetime.now().isoformat()
        
        logger.info(f"Analysis complete for job {job_id}: {summary['successful_reconstructions']} shots")
        
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {e}", exc_info=True)
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)


@app.post("/api/analyze/{job_id}")
async def start_analysis(
    job_id: str,
    background_tasks: BackgroundTasks,
    clip_duration: float = 1.0,
    device: str = "auto",
    multi_person: bool = True
):
    """Start shot detection and 3D reconstruction analysis"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job['status'] != 'uploaded':
        raise HTTPException(status_code=400, detail=f"Job is already {job['status']}")
    
    # Start background task
    background_tasks.add_task(
        run_analysis_task,
        job_id,
        clip_duration,
        device,
        multi_person
    )
    
    job['status'] = 'queued'
    
    return {
        'job_id': job_id,
        'status': 'queued',
        'message': 'Analysis started'
    }


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get analysis status"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    return {
        'job_id': job_id,
        'status': job['status'],
        'progress': job.get('progress', ''),
        'created_at': job.get('created_at'),
        'completed_at': job.get('completed_at'),
        'error': job.get('error')
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get complete analysis results"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job['status'] != 'complete':
        return {
            'job_id': job_id,
            'status': job['status'],
            'message': 'Analysis not complete'
        }
    
    return {
        'job_id': job_id,
        'status': job['status'],
        'results': job.get('results', {}),
        'filename': job['filename']
    }


@app.get("/api/shots/{job_id}")
async def get_shots_list(job_id: str):
    """Get list of detected shots with thumbnails"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job['status'] != 'complete':
        raise HTTPException(status_code=400, detail="Analysis not complete")
    
    results = job.get('results', {})
    shots = results.get('shots', [])
    
    # Format shots for frontend
    shots_list = []
    for shot in shots:
        shot_info = {
            'shot_number': shot['shot_number'],
            'frame': shot['shot_frame'],
            'time_s': shot['shot_time_s'],
            'success': shot.get('success', False),
            'clip_path': shot.get('clip_path'),
            'thumbnail_url': f"/api/shots/{job_id}/{shot['shot_number']}/thumbnail",
            'viewer_url': f"/view/{job_id}/{shot['shot_number']}"
        }
        
        # Add reconstruction info if available
        if shot.get('reconstruction'):
            recon = shot['reconstruction']
            shot_info['swing_analysis'] = recon.get('swing_analysis', {})
            shot_info['has_3d'] = recon.get('success', False)
        
        shots_list.append(shot_info)
    
    return {
        'job_id': job_id,
        'total_shots': len(shots_list),
        'successful_reconstructions': results.get('successful_reconstructions', 0),
        'shots': shots_list
    }


@app.get("/api/shots/{job_id}/{shot_number}/thumbnail")
async def get_shot_thumbnail(job_id: str, shot_number: int):
    """Get thumbnail for a specific shot"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    output_dir = Path(job['output_dir'])
    
    # Try to get first frame from swing_frames directory
    shot_dir = output_dir / f"shot_{shot_number:03d}" / "swing_frames"
    
    if shot_dir.exists():
        frames = sorted(shot_dir.glob("swing_*.jpg"))
        if frames:
            # Get middle frame as thumbnail
            mid_idx = len(frames) // 2
            return FileResponse(frames[mid_idx], media_type="image/jpeg")
    
    # Fallback: extract from clip video
    clips_dir = output_dir / "clips"
    clip_path = clips_dir / f"shot_{shot_number:03d}_clip.mp4"
    
    if clip_path.exists():
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Thumbnail not found")


@app.get("/api/shots/{job_id}/{shot_number}/3d")
async def get_shot_3d_data(job_id: str, shot_number: int):
    """Get 3D reconstruction data for a specific shot with multi-person support"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job['status'] != 'complete':
        raise HTTPException(status_code=400, detail="Analysis not complete")
    
    # Get shot details
    results = job.get('results', {})
    shots = results.get('shots', [])
    
    shot = None
    for s in shots:
        if s['shot_number'] == shot_number:
            shot = s
            break
    
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    
    # Get 3D models
    output_dir = Path(job['output_dir'])
    shot_dir = output_dir / f"shot_{shot_number:03d}"
    meshes_dir = shot_dir / "meshes_3d"
    
    if not meshes_dir.exists():
        raise HTTPException(status_code=404, detail="3D models not found")
    
    # List GLB files and parse multi-person format
    import re
    model_files = sorted(meshes_dir.glob("*.glb"))
    
    # Check if multi-person format: frame_NNNNNN_personX.glb
    multi_person_pattern = re.compile(r'frame_(\d+)_person(\d+)\.glb')
    is_multi_person = any(multi_person_pattern.match(m.name) for m in model_files)
    
    # Load metadata
    metadata_path = meshes_dir / "reconstruction_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Load clip metadata for homography
    clips_dir = output_dir.parent / "clips"
    clip_metadata_path = clips_dir / f"shot_{shot_number:03d}_clip.json"
    homography_matrix = None
    court_corners = None
    if clip_metadata_path.exists():
        with open(clip_metadata_path, 'r') as f:
            clip_meta = json.load(f)
            homography_matrix = clip_meta.get('homography_matrix')
            court_corners = clip_meta.get('court_corners')
    
    # Load analysis report
    report_path = shot_dir / "analysis_report.json"
    analysis = {}
    if report_path.exists():
        with open(report_path, 'r') as f:
            analysis = json.load(f)
    
    if is_multi_person:
        # Group models by frame
        frames_data = {}
        for model_file in model_files:
            match = multi_person_pattern.match(model_file.name)
            if match:
                frame_idx = int(match.group(1))
                person_id = int(match.group(2))
                
                if frame_idx not in frames_data:
                    frames_data[frame_idx] = []
                
                # Get bbox from metadata if available
                bbox = None
                if metadata and 'frames' in metadata:
                    for frame_meta in metadata['frames']:
                        if frame_meta.get('frame_index') == frame_idx:
                            if 'persons' in frame_meta:
                                for person_meta in frame_meta['persons']:
                                    if person_meta.get('person_id') == person_id:
                                        bbox = person_meta.get('bbox', [])
                                        break
                            break
                
                frames_data[frame_idx].append({
                    'person_id': person_id,
                    'model_url': f"/api/shots/{job_id}/{shot_number}/models/{model_file.name}",
                    'bbox': bbox,
                    'filename': model_file.name
                })
        
        # Sort frames and convert to list
        frames = []
        for frame_idx in sorted(frames_data.keys()):
            # Sort persons by ID
            persons = sorted(frames_data[frame_idx], key=lambda p: p['person_id'])
            frames.append({
                'frame_index': frame_idx,
                'persons': persons
            })
        
        return {
            'job_id': job_id,
            'shot_number': shot_number,
            'multi_person': True,
            'frames': frames,
            'num_frames': len(frames),
            'metadata': metadata,
            'analysis': analysis,
            'homography_matrix': homography_matrix,
            'court_corners': court_corners,
            'velocity_plot': f"/api/shots/{job_id}/{shot_number}/velocity_plot"
        }
    else:
        # Single-person format
        return {
            'job_id': job_id,
            'shot_number': shot_number,
            'multi_person': False,
            'models': [f"/api/shots/{job_id}/{shot_number}/models/{m.name}" for m in model_files],
            'num_frames': len(model_files),
            'metadata': metadata,
            'analysis': analysis,
            'homography_matrix': homography_matrix,
            'court_corners': court_corners,
            'velocity_plot': f"/api/shots/{job_id}/{shot_number}/velocity_plot"
        }


@app.get("/api/shots/{job_id}/{shot_number}/models/{filename}")
async def get_3d_model(job_id: str, shot_number: int, filename: str):
    """Serve 3D model file"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    output_dir = Path(job['output_dir'])
    
    model_path = output_dir / f"shot_{shot_number:03d}" / "meshes_3d" / filename
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(model_path, media_type="model/gltf-binary")


@app.get("/api/shots/{job_id}/{shot_number}/velocity_plot")
async def get_velocity_plot(job_id: str, shot_number: int):
    """Get velocity profile plot"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    output_dir = Path(job['output_dir'])
    
    plot_path = output_dir / f"shot_{shot_number:03d}" / "velocity_profile.png"
    
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(plot_path, media_type="image/png")


@app.get("/api/shots/{job_id}/{shot_number}/frames/{frame_index}")
async def get_shot_frame(job_id: str, shot_number: int, frame_index: int):
    """Get original video frame image for a specific shot and frame"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    output_dir = Path(job['output_dir'])
    
    # Try swing_frames directory first
    swing_frames_dir = output_dir / f"shot_{shot_number:03d}" / "swing_frames"
    
    if swing_frames_dir.exists():
        # Look for frame files (format: swing_XXXXXX.jpg or frame_XXXXXX.jpg)
        import re
        frame_files = sorted(swing_frames_dir.glob("*.jpg"))
        
        if frame_files:
            # Try to match frame index
            if frame_index < len(frame_files):
                return FileResponse(frame_files[frame_index], media_type="image/jpeg")
            
            # Try to find by frame number in filename
            for frame_file in frame_files:
                match = re.search(r'(\d+)', frame_file.stem)
                if match and int(match.group(1)) == frame_index:
                    return FileResponse(frame_file, media_type="image/jpeg")
    
    # Fallback: extract from clip video
    clips_dir = output_dir / "clips"
    clip_path = clips_dir / f"shot_{shot_number:03d}_clip.mp4"
    
    if clip_path.exists():
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                return Response(content=buffer.tobytes(), media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Frame not found")


@app.get("/api/results/{job_id}/full_video")
async def get_full_analysis_video(job_id: str):
    """Get full analysis video with overlays"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job['status'] != 'complete':
        raise HTTPException(status_code=400, detail="Analysis not complete")
    
    output_dir = Path(job['output_dir'])
    video_path = output_dir / "full_analysis.mp4"
    
    if not video_path.exists():
        logger.warning(f"Video not found at: {video_path}")
        raise HTTPException(status_code=404, detail=f"Full analysis video not found at {video_path}")
    
    logger.info(f"Serving video: {video_path} (size: {video_path.stat().st_size} bytes)")
    
    return FileResponse(
        video_path, 
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(video_path.stat().st_size),
            "Cache-Control": "public, max-age=3600"
        }
    )


@app.head("/api/results/{job_id}/full_video")
async def head_full_analysis_video(job_id: str):
    """Check if full analysis video exists (HEAD request)"""
    
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job['status'] != 'complete':
        raise HTTPException(status_code=400, detail="Analysis not complete")
    
    output_dir = Path(job['output_dir'])
    video_path = output_dir / "full_analysis.mp4"
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Full analysis video not found")
    
    # Return empty response with headers for HEAD request
    return Response(
        status_code=200,
        headers={
            "Content-Type": "video/mp4",
            "Accept-Ranges": "bytes",
            "Content-Length": str(video_path.stat().st_size)
        }
    )


# Mount frontend static files
frontend_dir = Path(__file__).parent / "frontend" / "build"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

