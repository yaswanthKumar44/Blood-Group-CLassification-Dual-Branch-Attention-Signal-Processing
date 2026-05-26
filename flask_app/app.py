"""
Blood Group Classification - Flask Web Application
Dual-Branch + Attention + Signal Processing Model
"""

import os
import io
import base64
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==============================================================================
# CONFIG
# ==============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # flask_app/
PARENT_DIR = os.path.dirname(BASE_DIR)                            # project root
MODEL_PATH = os.path.join(PARENT_DIR, "models", "best_dual_model.pth")

# Sorted class names — must match sorted(os.listdir(train/)) from training
CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# MODEL DEFINITION (mirrors training code exactly)
# ==============================================================================
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, f1, f2):
        weights = self.attn(torch.cat([f1, f2], dim=1))
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        return w1 * f1 + w2 * f2


class DualBranchNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.raw    = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=0)
        self.signal = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=0)
        dim         = self.raw.num_features
        self.fusion = AttentionFusion(dim)
        self.fc     = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x1, x2):
        f1    = self.raw(x1)
        f2    = self.signal(x2)
        fused = self.fusion(f1, f2)
        return self.fc(fused)


# ==============================================================================
# GRAD-CAM WRAPPER
# ==============================================================================
class GradCAMWrapper(nn.Module):
    def __init__(self, model, signal_tensor):
        super().__init__()
        self.model  = model
        self.signal = signal_tensor

    def forward(self, x):
        return self.model(x, self.signal)


# ==============================================================================
# SIGNAL PROCESSING (identical to training)
# ==============================================================================
def enhance_fingerprint(img: np.ndarray) -> np.ndarray:
    """CLAHE → Gaussian Blur → Sobel magnitude → RGB stack."""
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(gray)
    img   = cv2.GaussianBlur(img, (3, 3), 0)
    sx    = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sy    = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    img   = cv2.magnitude(sx, sy)
    img   = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img   = np.stack([img, img, img], axis=-1).astype(np.uint8)
    return img


# ==============================================================================
# INFERENCE TRANSFORMS (mirrors val_tfms in training)
# ==============================================================================
infer_tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])


# ==============================================================================
# LOAD MODEL ONCE AT STARTUP
# ==============================================================================
def load_model():
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = DualBranchNet(num_classes=len(CLASSES)).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Model loaded on {DEVICE}")
    return model


MODEL = load_model()


# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================
def predict(pil_image: Image.Image):
    """Return prediction dict including signal-processed and Grad-CAM images."""
    # Convert to numpy RGB
    original_arr = np.array(pil_image.convert("RGB"))

    # Signal processing
    signal_arr   = enhance_fingerprint(original_arr)

    # Transforms → tensors
    raw_tensor    = infer_tfms(image=original_arr)["image"].unsqueeze(0).to(DEVICE)
    signal_tensor = infer_tfms(image=signal_arr)["image"].unsqueeze(0).to(DEVICE)

    # ── Forward pass ──
    with torch.no_grad():
        out   = MODEL(raw_tensor, signal_tensor)
        probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()

    pred_idx    = int(probs.argmax())
    pred_label  = CLASSES[pred_idx]
    confidence  = float(probs[pred_idx])

    # ── Grad-CAM ──
    wrapped       = GradCAMWrapper(MODEL, signal_tensor)
    target_layer  = MODEL.raw.blocks[-1]
    cam           = GradCAM(model=wrapped, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=raw_tensor)[0]

    # Reconstruct float RGB for overlay
    raw_np = raw_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8)
    cam_img = show_cam_on_image(raw_np, grayscale_cam, use_rgb=True)

    # ── Encode images as base64 ──
    def arr_to_b64(arr: np.ndarray) -> str:
        img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Resize for display
    display_original = np.array(pil_image.convert("RGB").resize((224, 224)))

    return {
        "prediction":   pred_label,
        "confidence":   round(confidence * 100, 2),
        "probabilities": {cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, probs)},
        "original_img":  arr_to_b64(display_original),
        "signal_img":    arr_to_b64(cv2.resize(signal_arr, (224, 224))),
        "gradcam_img":   arr_to_b64(cam_img),
    }


# ==============================================================================
# FLASK APP
# ==============================================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.route("/")
def index():
    return render_template("index.html", device=str(DEVICE).upper())


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format '{ext}'"}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read()))
        t0      = time.time()
        result  = predict(pil_img)
        result["inference_ms"] = round((time.time() - t0) * 1000, 1)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "classes": CLASSES})


if __name__ == "__main__":
    print("=" * 60)
    print("  Blood Group Classifier — Flask App")
    print(f"  Device : {DEVICE}")
    print(f"  Classes: {CLASSES}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
