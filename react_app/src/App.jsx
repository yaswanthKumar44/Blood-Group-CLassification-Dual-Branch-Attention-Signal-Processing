import { useState } from 'react';

function App() {
  const [patientName, setPatientName] = useState('');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (selectedFile) => {
    if (selectedFile) {
      setFile(selectedFile);
      const url = URL.createObjectURL(selectedFile);
      setPreview(url);
    }
  };

  const predict = async () => {
    if (!file || !patientName) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_name', patientName);

    try {
      // Point this to your locally running Flask app
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to the server. Is Flask running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      {/* LEFT SECTION: Input */}
      <div className="section">
        <h1>Blood Group AI</h1>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <label>Patient Name</label>
          <input 
            type="text" 
            placeholder="Enter Patient Name..." 
            value={patientName}
            onChange={(e) => setPatientName(e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <label>Fingerprint Image</label>
          <div 
            className="file-upload"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            {preview ? (
               <img src={preview} alt="Preview" className="preview-img" />
            ) : (
               <p style={{ color: 'var(--text-muted)' }}>Drag & Drop or Click to Upload Fingerprint</p>
            )}
            <input 
              id="file-input" 
              type="file" 
              accept="image/*" 
              style={{ display: 'none' }}
              onChange={(e) => handleFileChange(e.target.files[0])}
            />
          </div>
        </div>

        <button 
          onClick={predict} 
          disabled={!file || !patientName || loading}
        >
          {loading ? <div className="loader"></div> : 'RUN AI ANALYSIS'}
        </button>
        
        {error && <p style={{ color: '#ff3366', marginTop: '1rem', textAlign: 'center' }}>{error}</p>}
      </div>

      {/* RIGHT SECTION: Results */}
      <div className="section results-panel">
        <label>AI Diagnostics Result</label>
        
        {!result && !loading && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <p style={{ color: 'var(--text-muted)', textAlign: 'center' }}>
              Upload a fingerprint and run the analysis to view results and Grad-CAM activations.
            </p>
          </div>
        )}

        {loading && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
            <div className="loader" style={{ width: '50px', height: '50px', borderWidth: '5px' }}></div>
            <p style={{ color: 'var(--primary)' }}>Processing with Dual-Branch EfficientNetV2...</p>
          </div>
        )}

        {result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div style={{ textAlign: 'center' }}>
              <div className="prediction-badge">{result.prediction}</div>
              <p style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>
                Confidence: {result.confidence}%
              </p>
            </div>

            <div className="cam-grid">
              <div className="cam-img-wrap">
                <span>Signal Processed (CLAHE)</span>
                <img src={result.signal_img} alt="Signal Processing" />
              </div>
              <div className="cam-img-wrap">
                <span>Grad-CAM Activation</span>
                <img src={result.gradcam_img} alt="Grad-CAM" />
              </div>
            </div>
            
            <div style={{ marginTop: '1rem' }}>
               <label>Class Probabilities</label>
               <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                 {Object.entries(result.probabilities || {}).map(([cls, prob]) => (
                    <div key={cls} style={{ 
                      padding: '0.3rem 0.8rem', 
                      background: cls === result.prediction ? 'var(--primary)' : 'rgba(255,255,255,0.05)',
                      color: cls === result.prediction ? '#000' : 'var(--text-main)',
                      borderRadius: '5px',
                      fontSize: '0.85rem',
                      fontWeight: '700'
                    }}>
                      {cls}: {prob}%
                    </div>
                 ))}
               </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
