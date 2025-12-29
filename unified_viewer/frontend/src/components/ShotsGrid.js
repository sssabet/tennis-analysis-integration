import React from 'react';
import './ShotsGrid.css';
import AnalysisVideoPlayer from './AnalysisVideoPlayer';

function ShotsGrid({ jobId, shots, onShotSelect }) {
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${String(secs).padStart(4, '0')}`;
  };

  return (
    <div className="shots-grid-container">
      {jobId && <AnalysisVideoPlayer jobId={jobId} />}
      
      {shots.length === 0 ? (
        <div className="no-shots">
          <p>No shots detected in this video</p>
        </div>
      ) : (
        <div className="shots-grid">
          {shots.map((shot) => (
            <div
              key={shot.shot_number}
              className={`shot-card ${!shot.has_3d ? 'shot-card-no-3d' : ''}`}
              onClick={() => shot.has_3d && onShotSelect(shot)}
            >
              <div className="shot-thumbnail">
                <img
                  src={shot.thumbnail_url}
                  alt={`Shot ${shot.shot_number}`}
                  onError={(e) => {
                    e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="300" height="200"%3E%3Crect fill="%23ddd" width="300" height="200"/%3E%3Ctext x="50%25" y="50%25" font-size="20" text-anchor="middle" fill="%23999"%3ENo Preview%3C/text%3E%3C/svg%3E';
                  }}
                />
                {shot.has_3d && (
                  <div className="shot-3d-badge">
                    <span>🎨 3D Available</span>
                  </div>
                )}
              </div>
              
              <div className="shot-info">
                <h3>Shot #{shot.shot_number}</h3>
                <div className="shot-details">
                  <div className="detail-row">
                    <span className="label">Time:</span>
                    <span className="value">{formatTime(shot.time_s)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Frame:</span>
                    <span className="value">{shot.frame}</span>
                  </div>
                  
                  {shot.swing_analysis && (
                    <>
                      <div className="detail-row">
                        <span className="label">Max Velocity:</span>
                        <span className="value">
                          {shot.swing_analysis.max_velocity?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      {shot.swing_analysis.max_acceleration && (
                        <div className="detail-row">
                          <span className="label">Max Accel:</span>
                          <span className="value">
                            {shot.swing_analysis.max_acceleration.toFixed(2)}
                          </span>
                        </div>
                      )}
                    </>
                  )}
                </div>
                
                {shot.has_3d ? (
                  <button className="btn-view-3d">View 3D →</button>
                ) : (
                  <div className="no-3d-message">3D reconstruction failed</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ShotsGrid;

