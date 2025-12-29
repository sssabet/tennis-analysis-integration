import React, { useState } from 'react';
import './App.css';
import UploadSection from './components/UploadSection';
import ShotsGrid from './components/ShotsGrid';
import ShotViewer3D from './components/ShotViewer3D';

function App() {
  const [currentJob, setCurrentJob] = useState(null);
  const [shots, setShots] = useState([]);
  const [selectedShot, setSelectedShot] = useState(null);
  const [view, setView] = useState('upload'); // 'upload', 'shots', '3d'

  const handleAnalysisComplete = (jobId) => {
    setCurrentJob(jobId);
    setView('shots');
    
    // Fetch shots list
    fetch(`/api/shots/${jobId}`)
      .then(res => res.json())
      .then(data => {
        console.log('Shots loaded:', data);
        setShots(data.shots || []);
      })
      .catch(err => {
        console.error('Failed to fetch shots:', err);
      });
  };

  const handleShotSelect = (shot) => {
    setSelectedShot(shot);
    setView('3d');
  };

  const handleBackToShots = () => {
    setSelectedShot(null);
    setView('shots');
  };

  const handleBackToUpload = () => {
    setCurrentJob(null);
    setShots([]);
    setSelectedShot(null);
    setView('upload');
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎾 Tennis Shot Analysis & 3D Reconstruction</h1>
        <p className="subtitle">Detect shots with homography + View player movements in 3D</p>
      </header>
      
      <div className="App-container">
        {view === 'upload' && (
          <UploadSection onAnalysisComplete={handleAnalysisComplete} />
        )}
        
        {view === 'shots' && (
          <div className="shots-view">
            <div className="view-header">
              <h2>Detected Shots</h2>
              <button className="btn-secondary" onClick={handleBackToUpload}>
                ← New Video
              </button>
            </div>
            <ShotsGrid 
              jobId={currentJob}
              shots={shots}
              onShotSelect={handleShotSelect}
            />
          </div>
        )}
        
        {view === '3d' && selectedShot && (
          <div className="viewer-view">
            <div className="view-header">
              <h2>Shot #{selectedShot.shot_number} - 3D Reconstruction</h2>
              <button className="btn-secondary" onClick={handleBackToShots}>
                ← Back to Shots
              </button>
            </div>
            <ShotViewer3D 
              jobId={currentJob}
              shot={selectedShot}
            />
          </div>
        )}
      </div>
      
      <footer className="App-footer">
        <p>Unified Tennis Analysis System | Shot Detection + 3D Reconstruction</p>
      </footer>
    </div>
  );
}

export default App;


