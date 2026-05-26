"""
Blood Group Classification - Chaquopy Offline Predictor

This script is meant to be called directly from Java via Chaquopy.
It loads the PyTorch model once on cold start, then predicts on the given paths.
"""

import os
import io
import time
import base64
import warnings
import json
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Grad-CAM is optional depending on Chaquopy compilation success.
# If it fails to compile on Android, we can gracefully fallback.
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


# ==============================================================================
# CONFIG
# ==============================================================================
CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

# We use CPU primarily since Chaquopy/Android doesn't easily expose CUDA
DEVICE = torch.device("cpu")

# Model path must be mapped to Android internal storage or assets dir
# Chaquopy provides standard __file__ pointing to the android assets/python dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: You MUST copy 'best_dual_model.pth' into the android assets folder too!
MODEL_PATH = os.path.join(BASE_DIR, "best_dual_model.pth")


# ==============================================================================
# MODEL DEFINITION
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

class GradCAMWrapper(nn.Module):
    def __init__(self, model, signal_tensor):
        super().__init__()
        self.model  = model
        self.signal = signal_tensor

    def forward(self, x):
        return self.model(x, self.signal)

# ==============================================================================
# SIGNAL PROCESSING
# ==============================================================================
def enhance_fingerprint(img: np.ndarray) -> np.ndarray:
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

infer_tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# Global Instance
GLOBAL_MODEL = None

def load_model():
    global GLOBAL_MODEL
    if GLOBAL_MODEL is not None:
        return GLOBAL_MODEL
    
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = DualBranchNet(num_classes=len(CLASSES)).to(DEVICE)
    # CPU mapping is crucial for Android devices
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    GLOBAL_MODEL = model
    return model


# ==============================================================================
# MAIN PREDICTION ENTRYPOINT (Called by Java)
# ==============================================================================
def predict_image(image_path: str):
    """
    Entrypoint for Chaquopy.
    Reads an image from the Android storage, runs inference, returns JSON string.
    """
    try:
        model = load_model()
        pil_image = Image.open(image_path).convert("RGB")
        original_arr = np.array(pil_image)
        signal_arr   = enhance_fingerprint(original_arr)

        raw_tensor    = infer_tfms(image=original_arr)["image"].unsqueeze(0).to(DEVICE)
        signal_tensor = infer_tfms(image=signal_arr)["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out   = model(raw_tensor, signal_tensor)
            probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()

        pred_idx    = int(probs.argmax())
        pred_label  = CLASSES[pred_idx]
        confidence  = float(probs[pred_idx])

        result_dict = {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, probs)},
        }

        # Grad-CAM processing if installed
        if HAS_GRADCAM:
            wrapped = GradCAMWrapper(model, signal_tensor)
            target_layer = model.raw.blocks[-1]
            cam = GradCAM(model=wrapped, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=raw_tensor)[0]
            
            raw_np = raw_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8)
            cam_img = show_cam_on_image(raw_np, grayscale_cam, use_rgb=True)

            def arr_to_b64(arr: np.ndarray) -> str:
                img = Image.fromarray(arr.astype(np.uint8))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            result_dict["gradcam_img"] = arr_to_b64(cam_img)

        return json.dumps(result_dict)
    except Exception as e:
        return json.dumps({"error": str(e)})

