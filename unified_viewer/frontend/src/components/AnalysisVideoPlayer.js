import React, { useState, useEffect } from 'react';
import './AnalysisVideoPlayer.css';

function AnalysisVideoPlayer({ jobId }) {
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (!jobId) {
      setLoading(false);
      setError('No job ID provided');
      return;
    }

    // Check if video exists
    const videoSrc = `/api/results/${jobId}/full_video`;
    console.log('Checking for video at:', videoSrc);
    
    // Test if video is available - try HEAD first
    fetch(videoSrc, { method: 'HEAD' })
      .then(res => {
        console.log('HEAD response status:', res.status, res.statusText);
        if (res.ok) {
          console.log('Video found, setting URL');
          setVideoUrl(videoSrc);
          setError(null);
          setLoading(false);
        } else {
          // Try GET to see the actual error message
          console.log('HEAD failed, trying GET...');
          return fetch(videoSrc, { method: 'GET' })
            .then(getRes => {
              console.log('GET response status:', getRes.status);
              if (getRes.ok) {
                setVideoUrl(videoSrc);
                setError(null);
              } else {
                getRes.text().then(text => {
                  console.error('Video fetch error:', text);
                  setError(`Full analysis video not available (${getRes.status}: ${getRes.statusText})`);
                });
              }
              setLoading(false);
            });
        }
      })
      .catch(err => {
        console.error('Error checking video:', err);
        setError(`Failed to load analysis video: ${err.message}`);
        setLoading(false);
      });
  }, [jobId]);

  if (loading) {
    return (
      <div className="analysis-video-loading">
        <div className="spinner"></div>
        <p>Loading analysis video...</p>
      </div>
    );
  }

  if (error || !videoUrl) {
    return (
      <div className="analysis-video-error">
        <p>{error || 'Analysis video not available'}</p>
        <small>
          {error && error.includes('not available') 
            ? 'The full analysis video with overlays will be generated during processing.'
            : 'Please check the browser console for more details.'}
        </small>
        {jobId && (
          <button 
            onClick={() => {
              setLoading(true);
              setError(null);
              setVideoUrl(null);
              // Retry loading
              const videoSrc = `/api/results/${jobId}/full_video`;
              fetch(videoSrc, { method: 'HEAD' })
                .then(res => {
                  if (res.ok) {
                    setVideoUrl(videoSrc);
                    setError(null);
                  } else {
                    setError(`Video not available (${res.status})`);
                  }
                  setLoading(false);
                })
                .catch(err => {
                  setError(`Failed to load: ${err.message}`);
                  setLoading(false);
                });
            }}
            style={{ marginTop: '10px', padding: '8px 16px' }}
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div className={`analysis-video-container ${isExpanded ? 'expanded' : ''}`}>
      <div className="analysis-video-header">
        <h3>📹 Full Analysis Video</h3>
        <p className="video-description">
          Complete video with ball tracking, court detection, player positions, and shot overlays
        </p>
        <button 
          className="btn-toggle-expand"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? '▼ Collapse' : '▲ Expand'}
        </button>
      </div>
      
      <div className="analysis-video-wrapper">
        <video
          controls
          className="analysis-video"
          preload="metadata"
          src={videoUrl}
          key={videoUrl}
          playsInline
          onError={(e) => {
            console.error('Video load error:', e);
            const video = e.target;
            const error = video.error;
            if (error) {
              let errorMsg = 'Unknown error';
              switch(error.code) {
                case error.MEDIA_ERR_ABORTED:
                  errorMsg = 'Video loading aborted';
                  break;
                case error.MEDIA_ERR_NETWORK:
                  errorMsg = 'Network error while loading video';
                  break;
                case error.MEDIA_ERR_DECODE:
                  errorMsg = 'Video decoding error - file may be corrupted';
                  break;
                case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                  errorMsg = 'Video format not supported';
                  break;
              }
              console.error('Video error code:', error.code, errorMsg);
              console.error('Video error message:', error.message);
              setError(`Failed to load video: ${errorMsg}`);
            } else {
              setError('Failed to load video. Please check if the video file exists.');
            }
          }}
          onLoadedMetadata={(e) => {
            const video = e.target;
            console.log('Video metadata loaded successfully');
            console.log('Video duration:', video.duration, 'seconds');
            console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
          }}
          onCanPlay={() => {
            console.log('Video can play');
            setError(null);
          }}
          onLoadStart={() => {
            console.log('Video load started');
          }}
        >
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>
      
      <div className="analysis-video-info">
        <div className="info-item">
          <span className="info-label">Features:</span>
          <span className="info-value">
            Ball Tracking • Court Detection • Player Positions • Shot Events • Trajectories
          </span>
        </div>
      </div>
    </div>
  );
}

export default AnalysisVideoPlayer;


