"""
weapon_classifier.py

Train and run a small CNN to classify weapon HUD icons into categories:
    awp | rifle | smg | pistol | none

Pipeline
--------
1. Crop the slot-1 weapon icon ROI from full-frame screenshots
2. Convert to grayscale, resize to fixed dimensions
3. Augment from single-image samples (shift, brightness, contrast, noise)
4. Train a tiny CNN (2 conv + 2 FC layers)
5. At inference, crop same ROI -> predict category

Usage
-----
    python weapon_classifier.py train             # train from weapon_samples/
    python weapon_classifier.py predict <img>     # predict from a screenshot
    python weapon_classifier.py test              # quick sanity check on training images
"""

import sys
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = ["awp", "rifle", "smg", "pistol"]
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}

WEAPON_TO_CATEGORY = {
    # Snipers
    "awp":           "awp",
    "ssg_08":        "awp",
    # Rifles
    "ak-47":         "rifle",
    "m4a1-s":        "rifle",
    "m4a4":          "rifle",
    "galil_ar":      "rifle",
    "famas":         "rifle",
    "sg_553":        "rifle",
    # SMGs
    "mac-10":        "smg",
    "mp9":           "smg",
    "mp7":           "smg",
    # Pistols (no primary weapon in slot 1)
    "glock-18":      "pistol",
    "usp":           "pistol",
    "desert_eagle":  "pistol",
    "dual_berettas": "pistol",
    "p250":          "pistol",
    "five_seven":    "pistol",
    "tec-9":         "pistol",
}

# ---------------------------------------------------------------------------
# ROI definition — slot-1 weapon icon position (fractional coords)
# Calibrated from 1920x1080 screenshots (4:3 stretched on 16:9 monitor)
# ---------------------------------------------------------------------------

WEAPON_ROI = (0.695, 0.760, 0.855, 0.995)   # (y1, y2, x1, x2)
IMG_H, IMG_W = 32, 128   # resized input dimensions for CNN

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def crop_weapon_roi(frame: np.ndarray) -> np.ndarray:
    """Crop the slot-1 weapon icon region from a full game frame."""
    h, w = frame.shape[:2]
    y1, y2, x1, x2 = WEAPON_ROI
    return frame[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]


def preprocess(roi: np.ndarray) -> np.ndarray:
    """Convert ROI to the format expected by the CNN: grayscale, resized, float32."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    resized = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Data augmentation — generate many variants from a single image
# ---------------------------------------------------------------------------

def augment(roi: np.ndarray, n: int = 200) -> list[np.ndarray]:
    """
    Generate n augmented copies of a BGR ROI.

    Augmentations (applied randomly):
        - Brightness shift    ±40
        - Contrast scale      0.7–1.3
        - Gaussian noise      σ = 5–20
        - Horizontal shift    ±8 px
        - Vertical shift      ±4 px
        - Slight Gaussian blur (σ = 0–1.5)
    """
    results = []
    h, w = roi.shape[:2]

    for _ in range(n):
        img = roi.copy().astype(np.float32)

        # Brightness
        img += random.uniform(-40, 40)

        # Contrast
        mean = img.mean()
        img = (img - mean) * random.uniform(0.7, 1.3) + mean

        # Gaussian noise
        sigma = random.uniform(5, 20)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img += noise

        img = np.clip(img, 0, 255).astype(np.uint8)

        # Spatial shift
        dx = random.randint(-8, 8)
        dy = random.randint(-4, 4)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Slight blur
        if random.random() < 0.4:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        results.append(img)

    return results


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(samples_dir: str, augment_n: int = 200):
    """
    Load weapon sample screenshots, crop ROIs, augment, and return
    (X, y) arrays ready for training.

    X : np.ndarray, shape (N, 1, IMG_H, IMG_W), float32 [0, 1]
    y : np.ndarray, shape (N,), int64
    """
    samples_path = Path(samples_dir)
    X_list, y_list = [], []

    for img_file in sorted(samples_path.glob("*.png")):
        weapon_name = img_file.stem
        cat = WEAPON_TO_CATEGORY.get(weapon_name)
        if cat is None:
            print(f"  [SKIP] {img_file.name} — not in WEAPON_TO_CATEGORY")
            continue

        frame = cv2.imread(str(img_file))
        if frame is None:
            print(f"  [SKIP] {img_file.name} — could not read")
            continue

        roi = crop_weapon_roi(frame)
        cat_idx = CAT_TO_IDX[cat]

        # Original
        X_list.append(preprocess(roi))
        y_list.append(cat_idx)

        # Augmented copies
        for aug_roi in augment(roi, n=augment_n):
            X_list.append(preprocess(aug_roi))
            y_list.append(cat_idx)

        print(f"  {img_file.name:25s} → {cat:8s}  ({augment_n + 1} samples)")

    X = np.array(X_list, dtype=np.float32)[:, np.newaxis, :, :]   # (N, 1, H, W)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# CNN model (pure NumPy / PyTorch)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class WeaponCNN(nn.Module):
    """Tiny CNN for weapon icon classification."""

    def __init__(self, num_classes: int = len(CATEGORIES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "weapon_cnn.pth"


def train(samples_dir: str, epochs: int = 30, lr: float = 1e-3,
          augment_n: int = 200, batch_size: int = 32):
    if not HAS_TORCH:
        print("[ERROR] PyTorch is required.  pip install torch")
        sys.exit(1)

    print(f"Building dataset from {samples_dir} ...")
    X, y = build_dataset(samples_dir, augment_n=augment_n)
    print(f"Total samples: {len(y)}")
    for cat, idx in CAT_TO_IDX.items():
        print(f"  {cat:8s}: {(y == idx).sum()}")
    print()

    # Train / validation split (80/20, stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model = WeaponCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += len(yb)

        # --- validate ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(yb)

        train_acc = train_correct / train_total
        val_acc   = val_correct / val_total
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={train_loss/train_total:.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_model_cache: Optional["WeaponCNN"] = None


def load_model() -> "WeaponCNN":
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for weapon classification")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}")

    model = WeaponCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    _model_cache = model
    return model


def predict(frame: np.ndarray) -> Optional[str]:
    """
    Predict weapon category from a full game frame.

    Returns one of: 'awp', 'rifle', 'smg', 'pistol', or None if the
    weapon UI is not visible (model confidence too low).
    """
    roi = crop_weapon_roi(frame)
    x = preprocess(roi)
    tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :, :])

    model = load_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        idx = probs.argmax().item()
        conf = probs[idx].item()

    if conf < 0.4:
        return None
    return CATEGORIES[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    script_dir = Path(__file__).parent

    if cmd == "train":
        samples_dir = str(script_dir / "weapon_samples")
        train(samples_dir)

    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Usage: python weapon_classifier.py predict <image_path>")
            return
        frame = cv2.imread(sys.argv[2])
        if frame is None:
            print(f"Could not read {sys.argv[2]}")
            return
        result = predict(frame)
        print(f"Predicted: {result}")

    elif cmd == "test":
        model = load_model()
        samples_dir = script_dir / "weapon_samples"
        correct, total = 0, 0
        for img_file in sorted(samples_dir.glob("*.png")):
            weapon_name = img_file.stem
            expected = WEAPON_TO_CATEGORY.get(weapon_name)
            if expected is None:
                continue
            frame = cv2.imread(str(img_file))
            result = predict(frame)
            ok = (result == expected)
            correct += int(ok)
            total += 1
            print(f"  [{'PASS' if ok else 'FAIL'}] {img_file.name:25s} "
                  f"expected={expected:8s}  got={result}")
        print(f"\nAccuracy: {correct}/{total} ({correct/total:.1%})")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
