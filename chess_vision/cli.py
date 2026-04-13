"""CLI entry point for chess-vision."""

from pathlib import Path

import click

from chess_vision.config import DATA_DIR, MODELS_DIR


@click.group()
def main():
    """CNN-powered chess board recognition. Records OTB games as PGN."""
    pass


@main.command()
@click.option("--name", required=True, help="Name for this calibration profile.")
@click.option("--camera", default=0, help="Camera device index.")
def calibrate(name: str, camera: int):
    """Calibrate for a new chess set. Takes 2 photos of the starting position."""
    click.echo(f"Starting calibration for '{name}'...")
    click.echo("Not yet implemented.")


@main.command()
@click.option("--profile", required=True, help="Calibration profile name.")
@click.option("--output", default="game.pgn", help="Output PGN file path.")
@click.option("--camera", default=0, help="Camera device index.")
@click.option("--white", default="White", help="White player name.")
@click.option("--black", default="Black", help="Black player name.")
@click.option("--interval", default=0.5, help="Seconds between frame captures.")
def record(profile: str, output: str, camera: int, white: str, black: str, interval: float):
    """Record a chess game and save as PGN."""
    click.echo(f"Recording game with profile '{profile}'...")
    click.echo("Not yet implemented.")


@main.command()
@click.option("--stage", type=click.Choice(["base", "finetune"]), required=True)
@click.option("--epochs", default=20, help="Number of training epochs.")
@click.option("--lr", default=1e-3, help="Learning rate.")
@click.option("--batch-size", default=32, help="Batch size.")
def train(stage: str, epochs: int, lr: float, batch_size: int):
    """Train models on public datasets (base) or calibration data (finetune)."""
    if stage == "base":
        _train_base(epochs, lr, batch_size)
    else:
        click.echo("Use 'chess-vision calibrate' for fine-tuning.")


def _train_base(epochs: int, lr: float, batch_size: int):
    """Train base models on processed public datasets."""
    from torch.utils.data import DataLoader

    from chess_vision.models.occupancy import create_occupancy_model
    from chess_vision.models.piece import create_piece_model
    from chess_vision.models.export import export_to_onnx
    from chess_vision.training.dataset import create_occupancy_dataset, create_piece_dataset
    from chess_vision.training.augment import get_train_transforms, get_eval_transforms
    from chess_vision.training.train import train_model

    processed = DATA_DIR / "processed"
    checkpoints = MODELS_DIR / "checkpoints"

    train_tf = get_train_transforms()
    eval_tf = get_eval_transforms()

    # --- Occupancy model ---
    occ_dir = processed / "occupancy"
    if occ_dir.exists():
        click.echo("Training occupancy model...")
        occ_train = create_occupancy_dataset(occ_dir, "train", train_tf)
        occ_val = create_occupancy_dataset(occ_dir, "val", eval_tf)
        click.echo(f"  Train: {len(occ_train)} samples, Val: {len(occ_val)} samples")

        model = create_occupancy_model(pretrained=True)
        model = train_model(
            model,
            DataLoader(occ_train, batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(occ_val, batch_size=batch_size, num_workers=4),
            epochs=epochs,
            lr=lr,
            output_dir=checkpoints,
            model_name="occupancy",
        )
        export_to_onnx(model, MODELS_DIR / "occupancy.onnx")
        click.echo(f"  Exported to {MODELS_DIR / 'occupancy.onnx'}")
    else:
        click.echo(f"No occupancy data at {occ_dir}. Run scripts/download_data.py first.")

    # --- Piece model ---
    piece_dir = processed / "pieces"
    if piece_dir.exists():
        click.echo("Training piece model...")
        piece_train = create_piece_dataset(piece_dir, "train", train_tf)
        piece_val = create_piece_dataset(piece_dir, "val", eval_tf)
        click.echo(f"  Train: {len(piece_train)} samples, Val: {len(piece_val)} samples")

        model = create_piece_model(pretrained=True)
        model = train_model(
            model,
            DataLoader(piece_train, batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(piece_val, batch_size=batch_size, num_workers=4),
            epochs=epochs,
            lr=lr,
            output_dir=checkpoints,
            model_name="piece",
        )
        export_to_onnx(model, MODELS_DIR / "piece.onnx")
        click.echo(f"  Exported to {MODELS_DIR / 'piece.onnx'}")
    else:
        click.echo(f"No piece data at {piece_dir}. Run scripts/download_data.py first.")


@main.command()
@click.option("--image", required=True, type=click.Path(exists=True), help="Path to board image.")
@click.option("--profile", default=None, help="Calibration profile name.")
@click.option("--visualize", is_flag=True, help="Show annotated board visualization.")
def detect(image: str, profile: str | None, visualize: bool):
    """Detect board in a single image and print FEN."""
    click.echo(f"Detecting board in {image}...")
    click.echo("Not yet implemented.")


@main.command("list-profiles")
def list_profiles_cmd():
    """List all saved calibration profiles."""
    from chess_vision.calibration.profile import list_profiles

    profiles = list_profiles()
    if not profiles:
        click.echo("No calibration profiles found.")
        return
    for p in profiles:
        click.echo(f"  {p.name} (created {p.created_at})")


if __name__ == "__main__":
    main()
