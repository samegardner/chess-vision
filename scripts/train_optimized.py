"""Optimized training: feature extraction + linear classifier.

Instead of training the full CNN for 10 epochs through 300K images,
this script:
1. Pre-loads all images to RAM (~9GB)
2. Extracts features with a frozen MobileNetV3-Small backbone (one pass, ~3 min)
3. Trains linear classifiers on the features (~30 seconds)
4. Optionally fine-tunes the last block for a few epochs
5. Exports full models (backbone + classifier) to ONNX

Total time: ~5-10 minutes on Apple Silicon, no GPU required.

Usage:
    python scripts/train_optimized.py
    python scripts/train_optimized.py --finetune-epochs 5   # Also fine-tune last block
    python scripts/train_optimized.py --input-size 96       # Smaller input (faster)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.config import DATA_DIR, MODELS_DIR


def load_imagefolder_to_ram(
    root: Path, transform, desc: str = "Loading"
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load an entire ImageFolder dataset into RAM as tensors.

    Returns (images, labels, class_names).
    """
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    images = []
    labels = []
    for class_idx, class_dir in enumerate(class_dirs):
        paths = sorted(class_dir.glob("*.png"))
        for path in paths:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))
            labels.append(class_idx)

    print(f"  {desc}: {len(images)} images, {len(class_names)} classes")
    return torch.stack(images), torch.tensor(labels), class_names


class FeatureExtractor(nn.Module):
    """MobileNetV3-Small with classifier removed."""

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Keep everything except the classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten()
        # MobileNetV3-Small features output: 576 channels after avgpool
        self.feature_dim = 576

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class FullModel(nn.Module):
    """Complete model: MobileNetV3-Small backbone + linear classifier."""

    def __init__(self, feature_extractor: FeatureExtractor, classifier: nn.Linear):
        super().__init__()
        self.features = feature_extractor.features
        self.avgpool = feature_extractor.avgpool
        self.flatten = nn.Flatten()
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


@torch.no_grad()
def extract_features(
    extractor: FeatureExtractor,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Extract features from all images using the frozen backbone."""
    extractor.eval().to(device)
    features = []
    n_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="  Extracting features"):
        batch = images[i * batch_size : (i + 1) * batch_size].to(device)
        feat = extractor(batch)
        features.append(feat.cpu())

    return torch.cat(features)


def train_linear(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    features_val: torch.Tensor,
    labels_val: torch.Tensor,
    num_classes: int,
    feature_dim: int,
    epochs: int = 50,
    lr: float = 0.1,
    is_binary: bool = False,
) -> nn.Linear:
    """Train a linear classifier on pre-extracted features."""
    out_dim = 1 if is_binary else num_classes
    classifier = nn.Linear(feature_dim, out_dim)

    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=(len(features_train) + 4095) // 4096,
    )

    train_dataset = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        classifier.train()
        for feat, target in train_loader:
            if is_binary:
                pred = classifier(feat).squeeze(-1)
                loss = criterion(pred, target.float())
            else:
                pred = classifier(feat)
                loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Validation
        classifier.eval()
        with torch.no_grad():
            if is_binary:
                val_pred = classifier(features_val).squeeze(-1)
                val_acc = ((torch.sigmoid(val_pred) > 0.5).long() == labels_val).float().mean().item()
            else:
                val_pred = classifier(features_val)
                val_acc = (val_pred.argmax(dim=1) == labels_val).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs}: val_acc={val_acc:.4f} (best={best_acc:.4f})")

    classifier.load_state_dict(best_state)
    print(f"  Best val accuracy: {best_acc:.4f}")
    return classifier


def finetune_last_block(
    model: FullModel,
    images_train: torch.Tensor,
    labels_train: torch.Tensor,
    images_val: torch.Tensor,
    labels_val: torch.Tensor,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 128,
    is_binary: bool = False,
):
    """Fine-tune the last feature block + classifier for a few epochs."""
    # Freeze everything except last block + classifier
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last 2 feature blocks + classifier
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Fine-tuning {trainable:,}/{total:,} parameters ({trainable * 100 // total}%)")

    model = model.to(device)
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    train_dataset = TensorDataset(images_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(images_val, labels_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for imgs, targets in tqdm(train_loader, desc=f"  FT Epoch {epoch + 1}/{epochs}", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)
            if is_binary:
                out = out.squeeze(-1)
                loss = criterion(out, targets.float())
            else:
                loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Validation
        model.eval()
        correct = 0
        total_val = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)
                if is_binary:
                    preds = (torch.sigmoid(out.squeeze(-1)) > 0.5).long()
                else:
                    preds = out.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total_val += targets.size(0)

        val_acc = correct / total_val
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  FT Epoch {epoch + 1}/{epochs}: val_acc={val_acc:.4f} (best={best_acc:.4f})")

    model.load_state_dict(best_state)
    return model


def export_onnx(model: nn.Module, path: Path, input_size: int):
    """Export model to ONNX."""
    model.eval().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size)
    batch_dim = torch.export.Dim("batch")
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["output"],
        dynamic_shapes={"x": {0: batch_dim}},
    )
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  Exported: {path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Optimized chess-vision training")
    parser.add_argument("--input-size", type=int, default=100, help="Input image size (default: 100, native crop size)")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Fine-tuning epochs for last block (0 to skip)")
    parser.add_argument("--linear-epochs", type=int, default=50, help="Linear classifier training epochs")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    processed = DATA_DIR / "processed"
    input_size = args.input_size

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ===== OCCUPANCY MODEL =====
    print("\n===== OCCUPANCY MODEL =====")
    occ_dir = processed / "occupancy"
    if not occ_dir.exists():
        print(f"No data at {occ_dir}. Run scripts/download_data.py first.")
        return

    t0 = time.time()

    # Load data to RAM
    print("Loading training data to RAM...")
    occ_train_imgs, occ_train_labels, occ_classes = load_imagefolder_to_ram(
        occ_dir / "train", transform, "Train"
    )
    occ_val_imgs, occ_val_labels, _ = load_imagefolder_to_ram(
        occ_dir / "val", transform, "Val"
    )
    print(f"  Classes: {occ_classes}")

    # Extract features
    print("Extracting features with MobileNetV3-Small...")
    extractor = FeatureExtractor()
    occ_train_feats = extract_features(extractor, occ_train_imgs, device)
    occ_val_feats = extract_features(extractor, occ_val_imgs, device)

    # Train linear classifier
    print("Training linear classifier...")
    occ_classifier = train_linear(
        occ_train_feats, occ_train_labels,
        occ_val_feats, occ_val_labels,
        num_classes=2, feature_dim=extractor.feature_dim,
        epochs=args.linear_epochs, is_binary=True,
    )

    # Assemble and optionally fine-tune
    occ_model = FullModel(extractor, occ_classifier)
    if args.finetune_epochs > 0:
        print(f"Fine-tuning last block ({args.finetune_epochs} epochs)...")
        occ_model = finetune_last_block(
            occ_model, occ_train_imgs, occ_train_labels,
            occ_val_imgs, occ_val_labels,
            device, epochs=args.finetune_epochs, is_binary=True,
        )

    # Export
    export_onnx(occ_model, MODELS_DIR / "occupancy.onnx", input_size)
    # Save checkpoint for calibration fine-tuning
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "checkpoints").mkdir(exist_ok=True)
    torch.save(occ_model.state_dict(), MODELS_DIR / "checkpoints" / "occupancy_best.pt")

    occ_time = time.time() - t0
    print(f"Occupancy model done in {occ_time:.0f}s")

    # ===== PIECE MODEL =====
    print("\n===== PIECE MODEL =====")
    piece_dir = processed / "pieces"
    if not piece_dir.exists():
        print(f"No data at {piece_dir}.")
        return

    t0 = time.time()

    # Load data to RAM
    print("Loading training data to RAM...")
    piece_train_imgs, piece_train_labels, piece_classes = load_imagefolder_to_ram(
        piece_dir / "train", transform, "Train"
    )
    piece_val_imgs, piece_val_labels, _ = load_imagefolder_to_ram(
        piece_dir / "val", transform, "Val"
    )
    print(f"  Classes: {piece_classes}")

    # Extract features (reuse same backbone)
    print("Extracting features...")
    extractor2 = FeatureExtractor()  # Fresh copy
    piece_train_feats = extract_features(extractor2, piece_train_imgs, device)
    piece_val_feats = extract_features(extractor2, piece_val_imgs, device)

    # Train linear classifier
    print("Training linear classifier...")
    piece_classifier = train_linear(
        piece_train_feats, piece_train_labels,
        piece_val_feats, piece_val_labels,
        num_classes=len(piece_classes), feature_dim=extractor2.feature_dim,
        epochs=args.linear_epochs, is_binary=False,
    )

    # Assemble and optionally fine-tune
    piece_model = FullModel(extractor2, piece_classifier)
    if args.finetune_epochs > 0:
        print(f"Fine-tuning last block ({args.finetune_epochs} epochs)...")
        piece_model = finetune_last_block(
            piece_model, piece_train_imgs, piece_train_labels,
            piece_val_imgs, piece_val_labels,
            device, epochs=args.finetune_epochs, is_binary=False,
        )

    # Export
    export_onnx(piece_model, MODELS_DIR / "piece.onnx", input_size)
    torch.save(piece_model.state_dict(), MODELS_DIR / "checkpoints" / "piece_best.pt")

    piece_time = time.time() - t0
    print(f"Piece model done in {piece_time:.0f}s")

    print(f"\n===== TOTAL: {occ_time + piece_time:.0f}s =====")
    print(f"Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
