import React, { useState } from 'react';
import './UploadSection.css';

function UploadSection({ onAnalysisComplete }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState('');
  const [clipDuration, setClipDuration] = useState(1.0);
  const [multiPerson, setMultiPerson] = useState(true);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setProgress('Uploading video...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setUploading(false);
      
      // Start analysis
      startAnalysis(data.job_id);
      
    } catch (error) {
      console.error('Upload error:', error);
      setProgress('Upload failed: ' + error.message);
      setUploading(false);
    }
  };

  const startAnalysis = async (id) => {
    setAnalyzing(true);
    setProgress('Starting shot detection and 3D reconstruction...');

    try {
      const response = await fetch(`/api/analyze/${id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          clip_duration: clipDuration,
          device: 'auto',
          multi_person: multiPerson
        })
      });

      if (!response.ok) {
        throw new Error('Analysis start failed');
      }

      // Poll for status
      pollStatus(id);
      
    } catch (error) {
      console.error('Analysis error:', error);
      setProgress('Analysis failed: ' + error.message);
      setAnalyzing(false);
    }
  };

  const pollStatus = async (id) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/status/${id}`);
        const data = await response.json();
        
        setProgress(data.progress || data.status);
        
        if (data.status === 'complete') {
          clearInterval(interval);
          setAnalyzing(false);
          setProgress('Analysis complete!');
          setTimeout(() => {
            onAnalysisComplete(id);
          }, 500);
        } else if (data.status === 'failed') {
          clearInterval(interval);
          setAnalyzing(false);
          setProgress('Analysis failed: ' + (data.error || 'Unknown error'));
        }
      } catch (error) {
        console.error('Status poll error:', error);
      }
    }, 2000); // Poll every 2 seconds
  };

  return (
    <div className="upload-section">
      <div className="upload-card">
        <h2>Upload Tennis Video</h2>
        <p className="description">
          Upload a tennis match video to detect shots and view 3D player reconstructions
        </p>
        
        <div className="upload-area">
          <input
            type="file"
            accept="video/mp4,video/avi,video/mov,video/mkv"
            onChange={handleFileChange}
            disabled={uploading || analyzing}
            id="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            {file ? (
              <span>✓ {file.name}</span>
            ) : (
              <span>📹 Choose Video File</span>
            )}
          </label>
        </div>

        <div className="settings">
          <div className="setting-group">
            <label>
              Clip Duration (seconds):
              <input
                type="number"
                min="0.5"
                max="3.0"
                step="0.1"
                value={clipDuration}
                onChange={(e) => setClipDuration(parseFloat(e.target.value))}
                disabled={uploading || analyzing}
              />
            </label>
          </div>
          
          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={multiPerson}
                onChange={(e) => setMultiPerson(e.target.checked)}
                disabled={uploading || analyzing}
              />
              Multi-Person Mode
            </label>
          </div>
        </div>

        <button
          className="btn-primary upload-btn"
          onClick={handleUpload}
          disabled={!file || uploading || analyzing}
        >
          {uploading ? 'Uploading...' : analyzing ? 'Analyzing...' : 'Upload & Analyze'}
        </button>

        {progress && (
          <div className="progress-message">
            {analyzing && <div className="spinner"></div>}
            <p>{progress}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadSection;

