/* ═══════════════════════════════════════════════════════════════
   BLOOD GROUP AI — JavaScript App Logic
   ═══════════════════════════════════════════════════════════════ */

// ── Animated particle background ─────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  const ctx    = canvas.getContext('2d');
  let   W, H, particles;

  const COLORS = ['#ff4757', '#3d8ef8', '#9b59b6', '#1abc9c', '#f39c12'];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkParticle() {
    return {
      x:    Math.random() * W,
      y:    Math.random() * H,
      r:    Math.random() * 1.5 + 0.5,
      vx:   (Math.random() - 0.5) * 0.3,
      vy:   (Math.random() - 0.5) * 0.3,
      c:    COLORS[Math.floor(Math.random() * COLORS.length)],
      a:    Math.random() * 0.6 + 0.1,
    };
  }

  function init() {
    resize();
    particles = Array.from({ length: 120 }, mkParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.c + Math.floor(p.a * 255).toString(16).padStart(2, '0');
      ctx.fill();
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    });
    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx   = particles[i].x - particles[j].x;
        const dy   = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(255,71,87,${0.08 * (1 - dist / 100)})`;
          ctx.lineWidth   = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  init();
  draw();
  window.addEventListener('resize', () => { resize(); });
})();

// ════════════════════════════════════════════════════════════════
// UPLOAD LOGIC
// ════════════════════════════════════════════════════════════════
const fileInput     = document.getElementById('file-input');
const dropZone      = document.getElementById('drop-zone');
const browseBtn     = document.getElementById('browse-btn');
const analyzeBtn    = document.getElementById('analyze-btn');
const uploadIdle    = document.getElementById('upload-idle');
const uploadPreview = document.getElementById('upload-preview');
const previewImg    = document.getElementById('preview-img');
const previewName   = document.getElementById('preview-filename');
const previewSize   = document.getElementById('preview-size');
const changeBtn     = document.getElementById('change-btn');
const tryAgainBtn   = document.getElementById('try-again-btn');

let selectedFile = null;

function fmtSize(bytes) {
  if (bytes < 1024)       return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(2) + ' MB';
}

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  const url    = URL.createObjectURL(file);
  previewImg.src      = url;
  previewName.textContent = file.name.length > 28 ? file.name.slice(0, 25) + '…' : file.name;
  previewSize.textContent = fmtSize(file.size);
  uploadIdle.style.display    = 'none';
  uploadPreview.style.display = 'block';
  analyzeBtn.disabled = false;
}

function clearFile() {
  selectedFile        = null;
  fileInput.value     = '';
  previewImg.src      = '';
  uploadIdle.style.display    = 'block';
  uploadPreview.style.display = 'none';
  analyzeBtn.disabled = true;
}

browseBtn.addEventListener('click',  () => fileInput.click());
changeBtn.addEventListener('click',  () => { clearFile(); fileInput.click(); });
fileInput.addEventListener('change', e  => setFile(e.target.files[0]));

// Drag & drop
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop',      e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) setFile(f);
  else showToast('Please drop a valid image file.');
});
dropZone.addEventListener('click', e => {
  if (e.target === dropZone && !selectedFile) fileInput.click();
});

// ════════════════════════════════════════════════════════════════
// LOADING ANIMATION
// ════════════════════════════════════════════════════════════════
const loadingOverlay = document.getElementById('loading-overlay');
const loadingBar     = document.getElementById('loading-bar');
const steps = ['step-signal', 'step-dual', 'step-attention', 'step-gradcam'];
let loadTimer = null;

function startLoading() {
  loadingOverlay.style.display = 'flex';
  loadingBar.style.width       = '0%';
  steps.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
  });
  document.getElementById(steps[0]).classList.add('active');

  let progress  = 0;
  let stepIndex = 0;
  const durations = [25, 40, 60, 85]; // % at which each step completes

  loadTimer = setInterval(() => {
    progress += 0.8;
    loadingBar.style.width = Math.min(progress, 95) + '%';

    // Advance step visuals
    if (stepIndex < durations.length && progress >= durations[stepIndex]) {
      document.getElementById(steps[stepIndex]).classList.remove('active');
      document.getElementById(steps[stepIndex]).classList.add('done');
      stepIndex++;
      if (stepIndex < steps.length) {
        document.getElementById(steps[stepIndex]).classList.add('active');
      }
    }
  }, 80);
}

function stopLoading() {
  clearInterval(loadTimer);
  loadingBar.style.width = '100%';
  steps.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active');
    el.classList.add('done');
  });
  setTimeout(() => {
    loadingOverlay.style.display = 'none';
  }, 400);
}

// ════════════════════════════════════════════════════════════════
// ANALYZE BUTTON — API CALL
// ════════════════════════════════════════════════════════════════
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  startLoading();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();

    stopLoading();

    if (data.error) {
      showToast('⚠️ ' + data.error);
      return;
    }

    renderResults(data);
  } catch (err) {
    stopLoading();
    showToast('Network error — is the server running?');
    console.error(err);
  }
});

// Try again
tryAgainBtn.addEventListener('click', () => {
  document.getElementById('results-section').style.display = 'none';
  clearFile();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ════════════════════════════════════════════════════════════════
// RENDER RESULTS
// ════════════════════════════════════════════════════════════════
const CLASS_COLORS = {
  'A+':  '#ff4757', 'A-':  '#ff6b6b',
  'B+':  '#3d8ef8', 'B-':  '#74b9ff',
  'AB+': '#9b59b6', 'AB-': '#a29bfe',
  'O+':  '#1abc9c', 'O-':  '#55efc4',
};

function renderResults(data) {
  const section = document.getElementById('results-section');
  section.style.display = 'block';

  // ── Prediction banner ──
  document.getElementById('result-group').textContent            = data.prediction;
  document.getElementById('result-confidence-label').textContent = `Confidence: ${data.confidence}%`;
  document.getElementById('inference-time').textContent          = `⏱ ${data.inference_ms} ms`;

  // Confidence ring (stroke-dasharray circumference = 2π × 52 ≈ 326.7)
  const pct    = data.confidence / 100;
  const offset = 326.7 * (1 - pct);
  const ring   = document.getElementById('ring-fg');
  ring.style.strokeDashoffset = offset;

  // Inject gradient into SVG
  const ringSvg = document.querySelector('.confidence-ring');
  if (!ringSvg.querySelector('defs')) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
      <linearGradient id="ring-gradient" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%"   stop-color="#ff4757"/>
        <stop offset="100%" stop-color="#3d8ef8"/>
      </linearGradient>`;
    ringSvg.prepend(defs);
  }

  document.getElementById('ring-pct').textContent = data.confidence + '%';

  // ── Images ──
  document.getElementById('img-original').src = data.original_img;
  document.getElementById('img-signal').src   = data.signal_img;
  document.getElementById('img-gradcam').src  = data.gradcam_img;

  // ── Probability bars ──
  const grid = document.getElementById('proba-grid');
  grid.innerHTML = '';

  const sorted = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cls, pct]) => {
    const isTop   = cls === data.prediction;
    const color   = CLASS_COLORS[cls] || '#ff4757';
    const item    = document.createElement('div');
    item.className = 'proba-item';
    item.innerHTML = `
      <div class="proba-header">
        <span class="proba-label" style="color:${isTop ? color : 'inherit'}">${cls}${isTop ? ' ✓' : ''}</span>
        <span class="proba-pct">${pct}%</span>
      </div>
      <div class="proba-bar-bg">
        <div class="proba-bar-fill" id="bar-${cls.replace('+','p').replace('-','m')}"
             style="width:0%; background: ${isTop
               ? `linear-gradient(90deg, ${color}, ${color}cc)`
               : `rgba(255,255,255,0.15)`};"></div>
      </div>`;
    grid.appendChild(item);

    // Animate bar after mount
    requestAnimationFrame(() => {
      setTimeout(() => {
        const bar = document.getElementById(`bar-${cls.replace('+','p').replace('-','m')}`);
        if (bar) bar.style.width = pct + '%';
      }, 100);
    });
  });

  // ── Infer info ──
  document.getElementById('infer-info').innerHTML = `
    Prediction : ${data.prediction}<br>
    Confidence : ${data.confidence}%<br>
    Latency    : ${data.inference_ms} ms<br>
    Device     : {{ device }}
  `.replace('{{ device }}', document.getElementById('device-label').textContent);

  // Scroll to results
  setTimeout(() => {
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
}

// ════════════════════════════════════════════════════════════════
// TOAST
// ════════════════════════════════════════════════════════════════
function showToast(msg) {
  const old = document.querySelector('.toast');
  if (old) old.remove();
  const t = document.createElement('div');
  t.className   = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
