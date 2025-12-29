#!/usr/bin/env python3
"""
Integration Script: Shot Detection + 3D Reconstruction
Detects shots using homography detection, clips 1-second segments,
and sends them to 3D reconstruction pipeline.
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import logging

# Add both repos to path
sys.path.insert(0, str(Path(__file__).parent / "tennis_shot_homography_detection"))
sys.path.insert(0, str(Path(__file__).parent / "tennis_single_player"))

from TennisAnalyzer import TennisAnalyzer
from main import TennisSwingAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """
    Resolve device string to a valid PyTorch device.
    
    Args:
        device: Device string ('auto', 'cuda', 'cpu', etc.)
        
    Returns:
        Resolved device string ('cuda' or 'cpu')
    """
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


class ShotTo3DIntegrator:
    """Integrates shot detection with 3D reconstruction"""
    
    def __init__(self, video_path: str, output_base_dir: Optional[Path] = None):
        """
        Initialize integrator
        
        Args:
            video_path: Path to input video
            output_base_dir: Base directory for outputs
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.output_base_dir = output_base_dir or Path(__file__).parent / "integrated_outputs"
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Video properties
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        logger.info(f"Initialized integrator for: {self.video_path}")
        logger.info(f"Video FPS: {self.fps}, Total frames: {self.total_frames}")
    
    def detect_shots(self, device: str = "auto") -> Tuple[List[Dict], Dict]:
        """
        Detect shots using the homography detection system
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
            
        Returns:
            Tuple of (shot_events list, homography_matrices dict)
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Detecting shots with homography...")
        logger.info("=" * 60)
        
        # Resolve device string (convert "auto" to "cuda" or "cpu")
        resolved_device = resolve_device(device)
        logger.info(f"Using device: {resolved_device}")
        
        # Initialize shot detector components
        base_dir = Path(__file__).parent / "tennis_shot_homography_detection"
        tracknet_weights = str(base_dir / "weights" / "tracknet.pth")
        courtside_weights = str(base_dir / "weights" / "courtside_yolo.pt")
        
        # Initialize ball tracker and court detector
        from BallTrackerFusion import BallTrackerFusion
        from TennisCourtKeypointDetector import CourtKeypointDetector
        
        ball_tracker = BallTrackerFusion(
            tracknet_weights=tracknet_weights,
            courtside_weights=courtside_weights,
            device=resolved_device,
        )
        
        court_model_path = str(base_dir / "weights" / "court_keypoints.pth")
        court_detector = CourtKeypointDetector(
            model_path=court_model_path, device=resolved_device
        )
        
        # Initialize audio analyzer
        from AudioAnalyzer import AudioAnalyzer
        audio_analyzer = AudioAnalyzer(str(self.video_path), fps=self.fps)
        audio_ok = audio_analyzer.extract_audio()
        if audio_ok:
            audio_analyzer.analyze_intensity()
            logger.info(f"Audio analysis complete")
        else:
            logger.warning("Audio analysis unavailable")
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        # Shot detection state
        shot_events = []
        homography_matrices = {}
        prev_corners = None
        homography_matrix = None
        prev_ball_court_y = None
        last_net_cross_frame = -10_000
        min_shot_gap_frames = 12
        last_shot_frame = -100
        shot_count = 0
        
        # Net line in court-map coordinates
        from CourtMapping import yOffset, courtHeight
        net_y = yOffset + int(courtHeight * 0.5)
        
        from CourtMapping import givePoint, courtMap
        
        # Audio peak detection helper
        def audio_peak_near(f_idx):
            if not audio_analyzer.intensity_per_frame:
                return False
            intensities = audio_analyzer.intensity_per_frame
            n = len(intensities)
            idx = max(0, min(n - 1, int(f_idx) - 1))
            lo = max(0, idx - 10)
            hi = min(n, idx + 11)
            if hi <= lo:
                return False
            return float(max(intensities[lo:hi])) >= 0.35
        
        frame_count = 0
        
        logger.info(f"Processing {self.total_frames} frames for shot detection...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect court
            corners, court_source = court_detector.detect(frame)
            if corners and all(c is not None for c in corners):
                prev_corners = corners
                try:
                    court_view, homography_matrix = courtMap(
                        frame, corners[0], corners[1], corners[2], corners[3]
                    )
                    homography_matrices[frame_count] = homography_matrix.copy()
                except Exception as e:
                    if prev_corners:
                        corners = prev_corners
            elif prev_corners:
                corners = prev_corners
                try:
                    court_view, homography_matrix = courtMap(
                        frame, corners[0], corners[1], corners[2], corners[3]
                    )
                    homography_matrices[frame_count] = homography_matrix.copy()
                except:
                    pass
            
            # Detect ball
            x, y, source = ball_tracker.detect_ball(frame)
            if x is not None and y is not None and homography_matrix is not None:
                try:
                    ball_court = givePoint(homography_matrix, (x, y))
                    
                    # Detect net crossing
                    if prev_ball_court_y is not None:
                        prev_side = prev_ball_court_y - net_y
                        cur_side = ball_court[1] - net_y
                        crossed = (prev_side == 0) or (cur_side == 0) or (prev_side * cur_side < 0)
                        moved_enough = abs(ball_court[1] - prev_ball_court_y) > 10
                        
                        if crossed and moved_enough and (frame_count - last_net_cross_frame) > 6:
                            last_net_cross_frame = frame_count
                            
                            # Check for audio peak nearby
                            if audio_peak_near(frame_count):
                                # This is a shot!
                                if (frame_count - last_shot_frame) > min_shot_gap_frames:
                                    last_shot_frame = frame_count
                                    shot_count += 1
                                    
                                    shot_event = {
                                        'frame': frame_count,
                                        'time_s': frame_count / self.fps,
                                        'shot_number': shot_count,
                                        'homography_matrix': homography_matrix.tolist() if homography_matrix is not None else None,
                                        'court_corners': [list(c) if c else None for c in corners] if corners else None,
                                        'ball_position_pixel': (x, y),
                                        'ball_position_court': (int(ball_court[0]), int(ball_court[1])),
                                    }
                                    shot_events.append(shot_event)
                                    logger.info(f"Shot #{shot_count} detected at frame {frame_count} ({shot_event['time_s']:.2f}s)")
                            
                            prev_ball_court_y = ball_court[1]
                        else:
                            if prev_ball_court_y is None:
                                prev_ball_court_y = ball_court[1]
                    else:
                        prev_ball_court_y = ball_court[1]
                except Exception as e:
                    pass
            
            if frame_count % 100 == 0:
                logger.info(f"  Processed {frame_count}/{self.total_frames} frames... (found {len(shot_events)} shots)")
        
        cap.release()
        
        logger.info(f"Detected {len(shot_events)} shots")
        return shot_events, homography_matrices
    
    def clip_video_segment(self, start_frame: int, end_frame: int, 
                          output_path: Path, homography_matrix: Optional[np.ndarray] = None,
                          court_corners: Optional[List] = None) -> bool:
        """
        Clip a video segment and save with homography metadata
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            output_path: Output video path
            homography_matrix: Homography matrix for this segment
            court_corners: Court corner coordinates
            
        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {self.video_path}")
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Extract frames
        for frame_idx in range(start_frame, min(end_frame + 1, self.total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                logger.warning(f"Could not read frame {frame_idx}")
        
        cap.release()
        out.release()
        
        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            'source_video': str(self.video_path),
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time_s': start_frame / fps,
            'end_time_s': end_frame / fps,
            'duration_s': (end_frame - start_frame) / fps,
            'fps': fps,
            'width': width,
            'height': height,
            'homography_matrix': homography_matrix.tolist() if homography_matrix is not None else None,
            'court_corners': court_corners,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved clip: {output_path} ({metadata['duration_s']:.2f}s)")
        return True
    
    def reconstruct_3d(self, clip_path: Path, metadata_path: Path, 
                      shot_number: int, multi_person: bool = True) -> Dict:
        """
        Run 3D reconstruction on a video clip
        
        Args:
            clip_path: Path to video clip
            metadata_path: Path to metadata JSON
            shot_number: Shot number for organization
            multi_person: Enable multi-person mode
            
        Returns:
            Reconstruction results
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STEP 2: 3D Reconstruction for Shot #{shot_number}")
        logger.info(f"{'=' * 60}")
        
        # Create output directory for this shot
        output_dir = self.output_base_dir / f"shot_{shot_number:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize 3D analyzer
        analyzer = TennisSwingAnalyzer(
            str(clip_path),
            output_dir=output_dir,
            multi_person=multi_person
        )
        
        # Run full pipeline
        result = analyzer.run_full_pipeline(
            side='right',  # Default to right arm
            extract_fps=None,  # Use video FPS
            target_person_id=None  # Auto-select person
        )
        
        # Add homography metadata to results
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                clip_metadata = json.load(f)
            result['homography_metadata'] = clip_metadata
        
        return result
    
    def generate_full_analysis_video(self, device: str = "auto") -> Optional[Path]:
        """
        Generate full analysis video with overlays using TennisAnalyzer
        
        Args:
            device: Device to use for processing
            
        Returns:
            Path to output video if successful, None otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("Generating full analysis video with overlays...")
            logger.info("=" * 60)
            
            resolved_device = resolve_device(device)
            
            # Initialize TennisAnalyzer
            base_dir = Path(__file__).parent / "tennis_shot_homography_detection"
            tracknet_weights = str(base_dir / "weights" / "tracknet.pth")
            courtside_weights = str(base_dir / "weights" / "courtside_yolo.pt")
            
            analyzer = TennisAnalyzer(
                tracknet_weights=tracknet_weights,
                courtside_weights=courtside_weights,
                device=resolved_device,
                show_court_overlay=True,
                show_ball=True,
                show_players=True,
                show_trajectory=True,
                show_pose_skeleton=True,
                show_court_keypoints=True,
            )
            
            # Output path
            output_path = self.output_base_dir / "full_analysis.mp4"
            
            # Process video
            stats = analyzer.process_video(
                input_path=str(self.video_path),
                output_path=str(output_path),
                show_display=False
            )
            
            logger.info(f"Full analysis video saved to: {output_path}")
            logger.info(f"Processed {stats.get('processed_frames', 0)} frames")
            
            return output_path if output_path.exists() else None
            
        except Exception as e:
            logger.warning(f"Failed to generate full analysis video: {e}")
            return None
    
    def process_all_shots(self, clip_duration_s: float = 1.0, 
                         device: str = "auto", multi_person: bool = True,
                         generate_full_video: bool = True) -> Dict:
        """
        Process all detected shots: clip and reconstruct
        
        Args:
            clip_duration_s: Duration of clip in seconds (centered on shot)
            device: Device for shot detection
            multi_person: Enable multi-person 3D reconstruction
            generate_full_video: Whether to generate full analysis video with overlays
            
        Returns:
            Summary of all processing
        """
        # Step 1: Detect shots
        shot_events, homography_matrices = self.detect_shots(device=device)
        
        if not shot_events:
            logger.warning("No shots detected! Cannot proceed with 3D reconstruction.")
            return {'success': False, 'error': 'No shots detected', 'shots': []}
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {len(shot_events)} shots for 3D reconstruction...")
        logger.info(f"{'=' * 60}")
        
        # Step 2: Clip and reconstruct each shot
        results = []
        clips_dir = self.output_base_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        for shot_idx, shot_event in enumerate(shot_events, 1):
            shot_frame = shot_event['frame']
            half_duration_frames = int((clip_duration_s / 2) * self.fps)
            
            start_frame = max(0, shot_frame - half_duration_frames)
            end_frame = min(self.total_frames - 1, shot_frame + half_duration_frames)
            
            # Get homography for this shot (use the shot frame's homography)
            homography = None
            if shot_frame in homography_matrices:
                homography = np.array(homography_matrices[shot_frame])
            elif shot_event.get('homography_matrix'):
                homography = np.array(shot_event['homography_matrix'])
            
            # Clip video
            clip_path = clips_dir / f"shot_{shot_idx:03d}_clip.mp4"
            success = self.clip_video_segment(
                start_frame=start_frame,
                end_frame=end_frame,
                output_path=clip_path,
                homography_matrix=homography,
                court_corners=shot_event.get('court_corners')
            )
            
            if not success:
                logger.error(f"Failed to clip shot #{shot_idx}")
                continue
            
            metadata_path = clip_path.with_suffix('.json')
            
            # Run 3D reconstruction
            try:
                recon_result = self.reconstruct_3d(
                    clip_path=clip_path,
                    metadata_path=metadata_path,
                    shot_number=shot_idx,
                    multi_person=multi_person
                )
                
                results.append({
                    'shot_number': shot_idx,
                    'shot_frame': shot_frame,
                    'shot_time_s': shot_event['time_s'],
                    'clip_path': str(clip_path),
                    'metadata_path': str(metadata_path),
                    'reconstruction': recon_result,
                    'success': recon_result.get('success', False)
                })
                
                logger.info(f"✓ Shot #{shot_idx} processed successfully")
                
            except Exception as e:
                logger.error(f"✗ Shot #{shot_idx} reconstruction failed: {e}", exc_info=True)
                results.append({
                    'shot_number': shot_idx,
                    'shot_frame': shot_frame,
                    'shot_time_s': shot_event['time_s'],
                    'clip_path': str(clip_path),
                    'metadata_path': str(metadata_path),
                    'success': False,
                    'error': str(e)
                })
        
        # Step 3: Generate full analysis video (optional)
        full_video_path = None
        if generate_full_video:
            full_video_path = self.generate_full_analysis_video(device=device)
        
        # Save summary
        summary = {
            'source_video': str(self.video_path),
            'total_shots_detected': len(shot_events),
            'total_shots_processed': len(results),
            'successful_reconstructions': sum(1 for r in results if r.get('success')),
            'fps': self.fps,
            'total_frames': self.total_frames,
            'full_analysis_video': str(full_video_path) if full_video_path else None,
            'shots': results
        }
        
        summary_path = self.output_base_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'=' * 60}")
        logger.info("PROCESSING COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Shots detected: {len(shot_events)}")
        logger.info(f"Shots processed: {len(results)}")
        logger.info(f"Successful: {sum(1 for r in results if r.get('success'))}")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"Results directory: {self.output_base_dir}")
        
        return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Integrate shot detection with 3D reconstruction'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input tennis video'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--clip-duration',
        type=float,
        default=1.0,
        help='Duration of clip in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device for shot detection (default: auto)'
    )
    parser.add_argument(
        '--multi-person',
        action='store_true',
        default=True,
        help='Enable multi-person 3D reconstruction (default: True)'
    )
    parser.add_argument(
        '--single-person',
        action='store_true',
        help='Force single-person mode'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.video_path).exists():
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Determine multi-person mode
    multi_person = not args.single_person if args.single_person else args.multi_person
    
    # Run integration
    try:
        integrator = ShotTo3DIntegrator(
            video_path=args.video_path,
            output_base_dir=Path(args.output_dir) if args.output_dir else None
        )
        
        summary = integrator.process_all_shots(
            clip_duration_s=args.clip_duration,
            device=args.device,
            multi_person=multi_person
        )
        
        if not summary.get('successful_reconstructions', 0):
            logger.warning("No successful reconstructions!")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Integration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

