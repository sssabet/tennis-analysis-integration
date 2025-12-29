import React, { useState, useEffect, Suspense, useRef } from 'react';
import { Canvas, useLoader, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Grid } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as THREE from 'three';
import './ShotViewer3D.css';

function Model3D({ url, position = [0, 0, 0], color = null, personId = 0 }) {
  const gltf = useLoader(GLTFLoader, url);
  
  if (!gltf || !gltf.scene) {
    return null;
  }

  // Clone the scene to avoid sharing materials between instances
  const scene = gltf.scene.clone();
  
  // Fix orientation: SAM3D models are typically upside down
  // Rotate 180 degrees around X-axis to flip them right-side up
  scene.rotation.x = Math.PI;
  
  // Scale the model appropriately (SAM3D models are in meters, roughly 1.7m tall)
  const scale = 1.0;
  scene.scale.set(scale, scale, scale);
  
  // Apply color tint if specified (for multi-person differentiation)
  if (color) {
    scene.traverse((child) => {
      if (child.isMesh && child.material) {
        const material = child.material.clone();
        // Apply a subtle tint by mixing with the color
        const originalColor = material.color.clone();
        material.color = new THREE.Color(color);
        material.emissive = originalColor.clone().multiplyScalar(0.1);
        child.material = material;
      }
    });
  }

  // Group to handle positioning and rotation
  return (
    <group position={position}>
      <primitive object={scene} />
    </group>
  );
}

// Court dimensions in meters (standard tennis singles court)
const COURT_LENGTH = 23.77; // 78 feet
const COURT_WIDTH = 10.97;  // 36 feet (singles)

// WASD Camera Controller Component
function WASDCameraController() {
  const { camera } = useThree();
  const moveState = useRef({ forward: false, backward: false, left: false, right: false });
  const velocity = useRef(new THREE.Vector3());
  const moveSpeed = 0.1;
  const damping = 0.85;

  useEffect(() => {
    const handleKeyDown = (e) => {
      const key = e.key.toLowerCase();
      switch(key) {
        case 'w':
        case 'arrowup':
          moveState.current.forward = true;
          e.preventDefault();
          break;
        case 's':
        case 'arrowdown':
          moveState.current.backward = true;
          e.preventDefault();
          break;
        case 'a':
        case 'arrowleft':
          moveState.current.left = true;
          e.preventDefault();
          break;
        case 'd':
        case 'arrowright':
          moveState.current.right = true;
          e.preventDefault();
          break;
      }
    };

    const handleKeyUp = (e) => {
      const key = e.key.toLowerCase();
      switch(key) {
        case 'w':
        case 'arrowup':
          moveState.current.forward = false;
          break;
        case 's':
        case 'arrowdown':
          moveState.current.backward = false;
          break;
        case 'a':
        case 'arrowleft':
          moveState.current.left = false;
          break;
        case 'd':
        case 'arrowright':
          moveState.current.right = false;
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  useFrame(() => {
    const direction = new THREE.Vector3();
    const frontVector = new THREE.Vector3(0, 0, -1);
    const sideVector = new THREE.Vector3(1, 0, 0);

    // Get camera direction
    frontVector.applyQuaternion(camera.quaternion);
    sideVector.applyQuaternion(camera.quaternion);
    
    // Remove Y component for horizontal movement
    frontVector.y = 0;
    sideVector.y = 0;
    frontVector.normalize();
    sideVector.normalize();

    // Calculate movement direction
    if (moveState.current.forward) direction.add(frontVector);
    if (moveState.current.backward) direction.sub(frontVector);
    if (moveState.current.left) direction.sub(sideVector);
    if (moveState.current.right) direction.add(sideVector);

    // Apply movement
    if (direction.length() > 0) {
      direction.normalize();
      velocity.current.add(direction.multiplyScalar(moveSpeed));
    }

    // Apply velocity with damping
    camera.position.add(velocity.current);
    velocity.current.multiplyScalar(damping);

    // Limit movement to reasonable bounds
    camera.position.y = Math.max(1, Math.min(30, camera.position.y));
  });

  return null;
}

function CourtGrid() {
  return (
    <>
      {/* Court surface - placed below the players */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]} receiveShadow>
        <planeGeometry args={[COURT_WIDTH, COURT_LENGTH]} />
        <meshStandardMaterial 
          color="#2a5d3f" 
          opacity={0.9} 
          transparent 
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Court lines */}
      <Grid
        args={[COURT_WIDTH, COURT_LENGTH, 8, 16]}
        cellSize={COURT_LENGTH / 16}
        cellThickness={1}
        cellColor="#ffffff"
        sectionSize={COURT_WIDTH / 8}
        sectionThickness={1.5}
        sectionColor="#ffffff"
        fadeDistance={50}
        fadeStrength={1}
        followCamera={false}
        infiniteGrid={false}
        position={[0, 0, 0]}
      />
      
      {/* Reference axes for debugging orientation */}
      <axesHelper args={[2]} position={[0, 0.1, 0]} />
    </>
  );
}

function ShotViewer3D({ jobId, shot }) {
  const [modelData, setModelData] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [loading, setLoading] = useState(true);
  const [velocityPlotUrl, setVelocityPlotUrl] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [originalFrameUrl, setOriginalFrameUrl] = useState(null);

  useEffect(() => {
    // Fetch 3D data for this shot
    fetch(`/api/shots/${jobId}/${shot.shot_number}/3d`)
      .then(res => res.json())
      .then(data => {
        console.log('3D data loaded:', data);
        setModelData(data);
        setVelocityPlotUrl(data.velocity_plot);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch 3D data:', err);
        setLoading(false);
      });
  }, [jobId, shot]);

  const handleFrameChange = (e) => {
    setCurrentFrame(parseInt(e.target.value));
  };

  // Auto-playback effect
  useEffect(() => {
    if (!isPlaying) return;
    
    const totalFrames = modelData ? (modelData.multi_person ? modelData.frames.length : modelData.models.length) : 0;
    if (totalFrames === 0) return;
    
    const baseFps = 30;
    const intervalMs = (1000 / baseFps) / playbackSpeed;
    
    const intervalId = setInterval(() => {
      setCurrentFrame(prev => {
        const next = prev + 1;
        if (next >= totalFrames) {
          setIsPlaying(false);
          return totalFrames - 1;
        }
        return next;
      });
    }, intervalMs);
    
    return () => clearInterval(intervalId);
  }, [isPlaying, currentFrame, modelData, playbackSpeed]);

  // Update original frame URL when frame changes
  useEffect(() => {
    if (modelData && jobId && shot) {
      const frameUrl = `/api/shots/${jobId}/${shot.shot_number}/frames/${currentFrame}`;
      setOriginalFrameUrl(frameUrl);
    }
  }, [currentFrame, modelData, jobId, shot]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handlePrevFrame = () => {
    if (!isPlaying) {
      setCurrentFrame(prev => Math.max(0, prev - 1));
    }
  };

  const handleNextFrame = () => {
    if (!isPlaying) {
      const totalFrames = modelData ? (modelData.multi_person ? modelData.frames.length : modelData.models.length) : 0;
      setCurrentFrame(prev => Math.min(totalFrames - 1, prev + 1));
    }
  };

  // Calculate court positions from bounding boxes
  const calculateCourtPosition = (bbox, personId = 0) => {
    if (!bbox || bbox.length < 4) {
      // Spread players along the court if no bbox
      const spacing = 3;
      return [personId * spacing - spacing, 0, 0];
    }
    
    // bbox format: [x, y, width, height] in pixels
    // Center of bounding box in pixels (foot position is bottom-center)
    const centerX = bbox[0] + bbox[2] / 2;
    const bottomY = bbox[1] + bbox[3]; // Bottom of bbox (feet position)
    
    // Simple scaling to court coordinates (will be refined with homography)
    // For now, normalize to court dimensions
    // Assuming video is roughly 1920x1080 and centered on court
    const scaleX = COURT_WIDTH / 1920;
    const scaleZ = COURT_LENGTH / 1080;
    
    const courtX = (centerX - 960) * scaleX;  // Center at 0
    const courtZ = (bottomY - 540) * scaleZ;  // Center at 0
    
    // Y position is 0 (on the court surface)
    return [courtX, 0, courtZ];
  };

  if (loading) {
    return (
      <div className="viewer-3d-loading">
        <div className="spinner"></div>
        <p>Loading 3D reconstruction...</p>
      </div>
    );
  }

  if (!modelData) {
    return (
      <div className="viewer-3d-error">
        <p>No 3D data available for this shot</p>
        <small>3D reconstruction may have failed or is still processing.</small>
      </div>
    );
  }

  // Handle multi-person format
  const isMultiPerson = modelData.multi_person && modelData.frames;
  
  if (isMultiPerson) {
    if (!modelData.frames || modelData.frames.length === 0) {
      return (
        <div className="viewer-3d-error">
          <p>No frames available in multi-person data</p>
        </div>
      );
    }
    
    const maxFrames = modelData.frames.length;
    const safeFrameIndex = Math.min(currentFrame, maxFrames - 1);
    const currentFrameData = modelData.frames[safeFrameIndex];
    
    if (!currentFrameData || !currentFrameData.persons || currentFrameData.persons.length === 0) {
      return (
        <div className="viewer-3d-error">
          <p>No persons detected in frame {safeFrameIndex + 1}</p>
        </div>
      );
    }
  } else {
    // Single-person format
    if (!modelData.models || modelData.models.length === 0) {
      return (
        <div className="viewer-3d-error">
          <p>No 3D models available for this shot</p>
          <small>3D reconstruction may have failed or is still processing.</small>
        </div>
      );
    }
    
    const currentModelUrl = modelData.models[currentFrame];
    
    if (!currentModelUrl) {
      return (
        <div className="viewer-3d-error">
          <p>Invalid model URL for frame {currentFrame + 1}</p>
        </div>
      );
    }
  }

  // Colors for different players
  const playerColors = [
    '#ff6b6b',  // Red
    '#4dabf7',  // Blue
    '#51cf66',  // Green
    '#ffd43b',  // Yellow
    '#9775fa',  // Purple
  ];

  const totalFrames = modelData ? (isMultiPerson ? modelData.frames.length : modelData.models.length) : 0;

  return (
    <div className="shot-viewer-3d">
      <div className="viewer-content">
        <div className="viewer-split">
        {/* Original Frame Container */}
        <div className="original-frame-container">
          <div className="frame-label">Original Frame</div>
          {originalFrameUrl ? (
            <img 
              src={originalFrameUrl} 
              alt={`Frame ${currentFrame + 1}`}
              className="original-frame"
              onError={(e) => {
                e.target.style.display = 'none';
                const placeholder = e.target.parentElement.querySelector('.frame-placeholder');
                if (placeholder) placeholder.style.display = 'flex';
              }}
            />
          ) : null}
          <div className="frame-placeholder" style={{ display: originalFrameUrl ? 'none' : 'flex' }}>
            <div className="placeholder-icon">📷</div>
            <p>{originalFrameUrl ? 'Loading frame...' : 'No frame available'}</p>
          </div>
        </div>

        {/* 3D Canvas Container */}
        <div className="viewer-3d-canvas">
          <div className="frame-label">3D Reconstruction</div>
          <Canvas shadows gl={{ antialias: true, alpha: true }}>
            <PerspectiveCamera makeDefault position={[0, 12, 20]} fov={50} />
            <WASDCameraController />
            <OrbitControls 
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              minDistance={3}
              maxDistance={60}
              target={[0, 1, 0]}
              maxPolarAngle={Math.PI / 2 - 0.1}
            />
            
            <ambientLight intensity={0.6} />
            <directionalLight 
              position={[10, 10, 5]} 
              intensity={1} 
              castShadow
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
            />
            <directionalLight position={[-10, 10, -5]} intensity={0.5} />
            
            {/* Render court */}
            <CourtGrid />
            
            {/* Render players */}
            <Suspense fallback={
              <mesh>
                <boxGeometry args={[1, 2, 1]} />
                <meshStandardMaterial color="orange" />
              </mesh>
            }>
              {isMultiPerson ? (
                // Multi-person: render all persons in current frame
                modelData.frames[Math.min(currentFrame, modelData.frames.length - 1)].persons
                  .filter(person => {
                    // Filter out low-confidence detections or placeholder models
                    // Only show the main players (typically person_id 0 and 1)
                    return person.person_id < 5 && person.model_url; // Limit to first 5 detected
                  })
                  .map((person, idx) => {
                    const position = calculateCourtPosition(person.bbox, person.person_id);
                    const color = playerColors[person.person_id % playerColors.length];
                    
                    return (
                      <Model3D 
                        key={`${person.person_id}-${idx}`}
                        url={person.model_url} 
                        position={position}
                        color={color}
                        personId={person.person_id}
                      />
                    );
                  })
              ) : (
                // Single-person: render one model
                <Model3D 
                  url={modelData.models[currentFrame]} 
                  position={[0, 0, 0]}
                  personId={0}
                />
              )}
            </Suspense>
            
            <Environment preset="sunset" />
          </Canvas>
        </div>
        </div>
      </div>
      
      {/* Playback Controls */}
      <div className="viewer-controls">
        <button 
          onClick={handlePrevFrame}
          disabled={currentFrame === 0 || isPlaying}
          className="control-button"
        >
          ⏮️ Prev
        </button>
        
        <button 
          onClick={handlePlayPause}
          className="control-button play-button"
        >
          {isPlaying ? '⏸️ Pause' : '▶️ Play'}
        </button>
        
        <button 
          onClick={handleNextFrame}
          disabled={currentFrame === totalFrames - 1 || isPlaying}
          className="control-button"
        >
          Next ⏭️
        </button>
        
        <div className="frame-control">
          <label>
            Frame: {currentFrame + 1} / {totalFrames}
            {isPlaying && ` • Playing at ${playbackSpeed}x speed`}
          </label>
          <input
            type="range"
            min="0"
            max={totalFrames - 1}
            value={currentFrame}
            onChange={handleFrameChange}
            className="frame-slider"
            disabled={isPlaying}
          />
        </div>
        
        <div className="speed-control">
          <label>Speed:</label>
          <select 
            value={playbackSpeed} 
            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            disabled={isPlaying}
          >
            <option value="0.5">0.5x</option>
            <option value="1.0">1.0x</option>
            <option value="1.5">1.5x</option>
            <option value="2.0">2.0x</option>
          </select>
        </div>
      </div>
      
      {/* WASD Controls Help */}
      <div className="wasd-help">
        <div className="wasd-help-content">
          <strong>WASD Controls:</strong> W/↑ Forward • S/↓ Backward • A/← Left • D/→ Right
        </div>
      </div>
    </div>
  );
}

export default ShotViewer3D;

