"""Training loop for occupancy and piece classification models."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except layer4 and fc (final block + classification head)."""
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    output_dir: Path = Path("models/checkpoints"),
    model_name: str = "model",
    patience: int = 5,
    freeze: bool = False,
) -> nn.Module:
    """Train with Adam, cosine LR schedule, early stopping.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        lr: Learning rate.
        output_dir: Directory for saving checkpoints.
        model_name: Name prefix for checkpoint files.
        patience: Early stopping patience (epochs without improvement).
        freeze: If True, freeze early layers and only train layer4 + fc.

    Returns:
        Trained model with best validation weights loaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if freeze:
        freeze_backbone(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Frozen backbone: training {trainable:,}/{total:,} parameters ({trainable*100//total}%)")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Determine loss function from model output size
    # 1 output = binary (BCEWithLogitsLoss), >1 = multiclass (CrossEntropyLoss)
    sample_out = model.fc.out_features if hasattr(model, "fc") else None
    if sample_out == 1:
        criterion = nn.BCEWithLogitsLoss()
        is_binary = True
    else:
        criterion = nn.CrossEntropyLoss()
        is_binary = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    best_path = output_dir / f"{model_name}_best.pt"

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if is_binary:
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, labels.float())
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False
            ):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                if is_binary:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, labels.float())
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                val_loss += loss.item() * images.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        # Early stopping
        if epoch + 1 - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    # Load best weights
    model.load_state_dict(torch.load(best_path, weights_only=True))
    print(f"Loaded best model from epoch {best_epoch}")
    return model
