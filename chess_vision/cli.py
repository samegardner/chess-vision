"""CLI entry point for chess-vision."""

from pathlib import Path

import click


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
    click.echo(f"Training stage: {stage}")
    click.echo("Not yet implemented.")


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
