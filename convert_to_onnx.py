"""
Convert best_dual_model.pth → model.onnx for offline Android inference.
Run: python convert_to_onnx.py
"""
import os
import torch
import torch.nn as nn
import timm

# ── Same model definition as training ─────────────────────────────────────────
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


# ── Load model ────────────────────────────────────────────────────────────────
CLASSES   = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PTH_PATH  = os.path.join(BASE_DIR, "models", "best_dual_model.pth")
ONNX_PATH = os.path.join(BASE_DIR, "models", "model.onnx")

print(f"Loading model from: {PTH_PATH}")
model = DualBranchNet(num_classes=len(CLASSES))
state = torch.load(PTH_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()
print("Model loaded successfully.")

# ── Dummy inputs (batch=1, 3 channels, 224×224) ───────────────────────────────
dummy_raw    = torch.randn(1, 3, 224, 224)
dummy_signal = torch.randn(1, 3, 224, 224)

# ── Export to ONNX ────────────────────────────────────────────────────────────
print(f"Exporting to ONNX: {ONNX_PATH}")
torch.onnx.export(
    model,
    (dummy_raw, dummy_signal),
    ONNX_PATH,
    input_names=["raw_image", "signal_image"],
    output_names=["logits"],
    dynamic_axes={
        "raw_image":    {0: "batch"},
        "signal_image": {0: "batch"},
        "logits":       {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)

size_mb = os.path.getsize(ONNX_PATH) / 1e6
print(f"\n✅ model.onnx saved! Size: {size_mb:.1f} MB")
print("Next: bundle model.onnx into the React Native APK with onnxruntime-react-native")
