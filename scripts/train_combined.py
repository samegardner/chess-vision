"""Train on combined data: ChessReD (real boards) + your board photos.

Loads all training images from multiple directories into RAM,
trains MobileNetV3-Small with feature extraction + fine-tuning.

Usage:
    python scripts/train_combined.py
    python scripts/train_combined.py --epochs 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.config import MODELS_DIR

# All data sources
DATA_DIRS = [
    Path("data/chessred_squares"),   # ChessReD processed squares
    Path("data/my_board"),            # Your board photos
]


def load_all_data(data_dirs: list[Path], transform, max_per_dir: int = 0):
    """Load images from multiple ImageFolder-style directories."""
    # Canonical class ordering (alphabetical, matches ImageFolder)
    all_classes = [
        "black_bishop", "black_king", "black_knight", "black_pawn",
        "black_queen", "black_rook", "empty",
        "white_bishop", "white_king", "white_knight", "white_pawn",
        "white_queen", "white_rook",
    ]
    class_to_idx = {name: i for i, name in enumerate(all_classes)}

    images = []
    labels = []
    source_counts = {}

    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"  Skipping {data_dir} (not found)")
            continue

        count = 0
        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in class_to_idx:
                print(f"  Warning: unknown class '{class_name}' in {data_dir}")
                continue

            label = class_to_idx[class_name]
            paths = sorted(class_dir.glob("*.png"))

            for path in paths:
                img = Image.open(path).convert("RGB")
                images.append(transform(img))
                labels.append(label)
                count += 1

                if max_per_dir > 0 and count >= max_per_dir:
                    break
            if max_per_dir > 0 and count >= max_per_dir:
                break

        source_counts[str(data_dir)] = count
        print(f"  {data_dir}: {count} images")

    print(f"  Total: {len(images)} images, {len(all_classes)} classes")
    return torch.stack(images), torch.tensor(labels), all_classes, source_counts


def main():
    parser = argparse.ArgumentParser(description="Train on combined real board data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load all data
    print("\nLoading data...")
    images, labels, all_classes, counts = load_all_data(DATA_DIRS, transform)

    if len(images) == 0:
        print("No data found! Run process_chessred.py and/or collect_data.py first.")
        return

    # Separate piece classes from empty for the two models
    piece_classes = [c for c in all_classes if c != "empty"]
    piece_class_to_idx = {name: i for i, name in enumerate(piece_classes)}
    empty_idx = all_classes.index("empty")

    # Split into train/val (80/20)
    n_val = max(1, len(images) // 5)
    n_train = len(images) - n_val
    perm = torch.randperm(len(images))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    # ===== OCCUPANCY MODEL =====
    print("\n=== OCCUPANCY MODEL ===")
    occ_labels = (labels != empty_idx).long()

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Linear(576, 1))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        batch_perm = train_idx[torch.randperm(len(train_idx))]
        correct, total = 0, 0
        for i in range(0, len(batch_perm), args.batch_size):
            idx = batch_perm[i:i + args.batch_size]
            x = images[idx].to(device)
            y = occ_labels[idx].float().to(device)
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            correct += ((torch.sigmoid(out) > 0.5).long() == y.long()).sum().item()
            total += len(idx)

        model.eval()
        with torch.no_grad():
            vx = images[val_idx].to(device)
            vy = occ_labels[val_idx].float().to(device)
            vout = model(vx).squeeze(-1)
            val_acc = ((torch.sigmoid(vout) > 0.5).long() == vy.long()).float().mean().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{args.epochs}: train_acc={correct / total:.4f} val_acc={val_acc:.4f}")

    print(f"  Final: {val_acc:.4f}")

    # Export occupancy
    model.eval().cpu()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 100, 100)
    batch_dim = torch.export.Dim("batch")
    torch.onnx.export(model, dummy, str(MODELS_DIR / "occupancy.onnx"),
                       input_names=["input"], output_names=["output"],
                       dynamic_shapes={"x": {0: batch_dim}})
    print(f"  Saved {MODELS_DIR / 'occupancy.onnx'}")

    # ===== PIECE MODEL =====
    print("\n=== PIECE MODEL ===")
    # Filter to occupied squares only, remap labels to piece_classes order
    occ_mask = labels != empty_idx
    piece_images = images[occ_mask]

    # Remap labels: all_classes index -> piece_classes index
    piece_labels_raw = labels[occ_mask]
    piece_labels = torch.zeros(len(piece_labels_raw), dtype=torch.long)
    for i, label in enumerate(piece_labels_raw):
        class_name = all_classes[label.item()]
        piece_labels[i] = piece_class_to_idx[class_name]

    print(f"  Occupied squares: {len(piece_images)}")
    for cls_name in piece_classes:
        n = (piece_labels == piece_class_to_idx[cls_name]).sum().item()
        print(f"    {cls_name}: {n}")

    n_val_p = max(1, len(piece_images) // 5)
    perm_p = torch.randperm(len(piece_images))
    train_idx_p, val_idx_p = perm_p[:len(piece_images) - n_val_p], perm_p[len(piece_images) - n_val_p:]

    piece_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    piece_model.classifier = nn.Sequential(nn.Linear(576, len(piece_classes)))
    piece_model.to(device)

    optimizer_p = torch.optim.Adam(piece_model.parameters(), lr=args.lr)
    criterion_p = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        piece_model.train()
        batch_perm = train_idx_p[torch.randperm(len(train_idx_p))]
        correct, total = 0, 0
        for i in range(0, len(batch_perm), args.batch_size):
            idx = batch_perm[i:i + args.batch_size]
            x = piece_images[idx].to(device)
            y = piece_labels[idx].to(device)
            out = piece_model(x)
            loss = criterion_p(out, y)
            loss.backward()
            optimizer_p.step()
            optimizer_p.zero_grad(set_to_none=True)
            correct += (out.argmax(1) == y).sum().item()
            total += len(idx)

        piece_model.eval()
        with torch.no_grad():
            vx = piece_images[val_idx_p].to(device)
            vy = piece_labels[val_idx_p].to(device)
            vout = piece_model(vx)
            val_acc_p = (vout.argmax(1) == vy).float().mean().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{args.epochs}: train_acc={correct / total:.4f} val_acc={val_acc_p:.4f}")

    print(f"  Final: {val_acc_p:.4f}")

    # Export piece model
    piece_model.eval().cpu()
    torch.onnx.export(piece_model, dummy, str(MODELS_DIR / "piece.onnx"),
                       input_names=["input"], output_names=["output"],
                       dynamic_shapes={"x": {0: batch_dim}})
    print(f"  Saved {MODELS_DIR / 'piece.onnx'}")

    # Save class ordering
    json.dump(piece_classes, open(MODELS_DIR / "piece_classes.json", "w"))
    print(f"  Saved {MODELS_DIR / 'piece_classes.json'}")
    print(f"  Class order: {piece_classes}")

    print("\n=== DONE ===")
    print(f"Occupancy val acc: {val_acc:.1%}")
    print(f"Piece val acc: {val_acc_p:.1%}")


if __name__ == "__main__":
    main()
