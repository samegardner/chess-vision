"""CLI entry point for chess-vision."""

import time
from datetime import datetime
from pathlib import Path

import click

from chess_vision.config import DATA_DIR, MODELS_DIR, PROFILES_DIR


@click.group()
def main():
    """CNN-powered chess board recognition. Records OTB games as PGN."""
    pass


@main.command()
@click.option("--name", required=True, help="Name for this calibration profile.")
@click.option("--camera", default=0, help="Camera device index.")
def calibrate(name: str, camera: int):
    """Calibrate for a new chess set. Takes 2 photos of the starting position."""
    from chess_vision.inference.camera import Camera
    from chess_vision.calibration.capture import capture_calibration_photos
    from chess_vision.calibration.label import label_calibration_squares, save_calibration_squares
    from chess_vision.calibration.profile import CalibrationProfile, save_profile
    from chess_vision.training.finetune import finetune_from_calibration

    click.echo(f"Starting calibration for '{name}'...")

    # Step 1: Capture photos
    with Camera(device_index=camera) as cam:
        white_photo, black_photo = capture_calibration_photos(cam)

    # Step 2: Detect board and label squares
    click.echo("Detecting board and labeling squares...")
    labeled_squares = label_calibration_squares(white_photo, black_photo)
    click.echo(f"  Labeled {len(labeled_squares)} squares")

    # Step 3: Save square images
    profile_dir = PROFILES_DIR / name
    squares_dir = profile_dir / "squares"
    save_calibration_squares(labeled_squares, squares_dir)
    click.echo(f"  Saved square images to {squares_dir}")

    # Step 4: Fine-tune models
    base_occ = MODELS_DIR / "checkpoints" / "occupancy_best.pt"
    base_piece = MODELS_DIR / "checkpoints" / "piece_best.pt"

    if not base_occ.exists() or not base_piece.exists():
        click.echo(
            "Warning: Base model checkpoints not found. "
            "Run 'chess-vision train --stage base' first. "
            "Skipping fine-tuning, saving profile without custom models."
        )
        occ_model_path = ""
        piece_model_path = ""
    else:
        click.echo("Fine-tuning models on calibration data...")
        occ_onnx, piece_onnx = finetune_from_calibration(
            base_occ, base_piece, squares_dir, profile_dir,
        )
        occ_model_path = str(occ_onnx.relative_to(PROFILES_DIR.parent))
        piece_model_path = str(piece_onnx.relative_to(PROFILES_DIR.parent))

    # Step 5: Save profile
    profile = CalibrationProfile(
        name=name,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        occupancy_model=occ_model_path,
        piece_model=piece_model_path,
        square_images_dir=str(squares_dir.relative_to(PROFILES_DIR.parent)),
    )
    path = save_profile(profile)
    click.echo(f"Profile saved to {path}")


@main.command()
@click.option("--profile", required=True, help="Calibration profile name.")
@click.option("--output", default="game.pgn", help="Output PGN file path.")
@click.option("--camera", default=0, help="Camera device index.")
@click.option("--white", default="White", help="White player name.")
@click.option("--black", default="Black", help="Black player name.")
@click.option("--interval", default=0.5, help="Seconds between frame captures.")
@click.option("--auto-detect", is_flag=True, help="Use auto board detection instead of manual corner selection.")
def record(profile: str, output: str, camera: int, white: str, black: str, interval: float, auto_detect: bool):
    """Record a chess game and save as PGN."""
    from chess_vision.inference.camera import Camera
    from chess_vision.inference.onnx_runtime import ONNXClassifier
    from chess_vision.inference.classify import classify_board, board_to_fen
    from chess_vision.board.detect import detect_board, select_corners
    from chess_vision.board.warp import compute_homography, warp_board
    from chess_vision.board.squares import extract_squares, remap_board_state
    from chess_vision.game.state import GameState
    from chess_vision.game.moves import MoveDetector
    from chess_vision.game.rules import resolve_move, detect_orientation
    from chess_vision.game.pgn import generate_pgn, save_pgn
    from chess_vision.calibration.profile import load_profile
    from chess_vision.config import ROOT_DIR

    # Load profile and models
    prof = load_profile(profile)
    click.echo(f"Loaded profile '{prof.name}' (created {prof.created_at})")

    occ_path = ROOT_DIR / prof.occupancy_model if prof.occupancy_model else MODELS_DIR / "occupancy.onnx"
    piece_path = ROOT_DIR / prof.piece_model if prof.piece_model else MODELS_DIR / "piece.onnx"

    if not occ_path.exists() or not piece_path.exists():
        click.echo(f"Error: Model files not found at {occ_path} or {piece_path}")
        click.echo("Run 'chess-vision train --stage base' and/or 'chess-vision calibrate' first.")
        return

    occ_model = ONNXClassifier(occ_path)
    piece_model = ONNXClassifier(piece_path)

    game = GameState()
    detector = MoveDetector()
    cached_homography = None
    flipped = False  # True if board is oriented with black on bottom

    click.echo("Starting recording. Press Ctrl+C to stop and save PGN.")
    click.echo(f"  Camera: {camera}, Interval: {interval}s")
    click.echo(f"  Output: {output}")
    click.echo()

    with Camera(device_index=camera) as cam:
        frame = cam.capture()

        if auto_detect:
            click.echo("Auto-detecting board...")
            corners = detect_board(frame)
            if corners is None:
                click.echo("Auto-detection failed. Falling back to manual corner selection.")
                corners = select_corners(frame)
        else:
            click.echo("Select the 4 corners of the chessboard...")
            corners = select_corners(frame)

        cached_homography = compute_homography(corners)
        warped = warp_board(frame, cached_homography)
        squares = extract_squares(warped)
        board_state = classify_board(squares, occ_model, piece_model)

        try:
            orientation = detect_orientation(board_state)
            click.echo(f"  Orientation: {orientation}")
            if orientation == "white_top":
                flipped = True
                board_state = remap_board_state(board_state)
                click.echo("  Board is flipped (black nearest camera). Remapping squares.")
        except ValueError:
            click.echo("  Warning: Could not auto-detect orientation, assuming white on bottom")

        detector.set_initial_board(board_state)
        click.echo("  Board detected. Waiting for moves...\n")

        try:
            while not game.is_game_over():
                time.sleep(interval)

                try:
                    frame = cam.capture()
                except RuntimeError as e:
                    click.echo(f"  ! Camera error: {e}. Retrying...")
                    continue

                warped = warp_board(frame, cached_homography)
                squares = extract_squares(warped)
                board_state = classify_board(squares, occ_model, piece_model)
                if flipped:
                    board_state = remap_board_state(board_state)

                changed = detector.detect_change(board_state)
                if changed is None:
                    continue

                # Resolve the move
                move = resolve_move(changed, game.board, board_state)
                if move is None:
                    click.echo(f"  ? Unresolved change on squares: {changed}")
                    continue

                # Get SAN before applying (board.san requires the move to be legal now)
                san = game.board.san(move)
                game.apply_move(move)

                # Print move
                if game.whose_turn() == "white":
                    # Black just moved
                    click.echo(f"  {game.move_number() - 1}... {san}")
                else:
                    # White just moved
                    click.echo(f"  {game.move_number()}. {san}")

        except KeyboardInterrupt:
            click.echo("\n\nRecording stopped.")
        finally:
            # Always save PGN, even on error
            result = game.result()
            pgn_str = generate_pgn(
                game.move_history,
                white_name=white,
                black_name=black,
                result=result,
            )
            save_pgn(pgn_str, Path(output))
            click.echo(f"\nGame saved to {output}")
            click.echo(f"  Moves: {len(game.move_history)}")
            click.echo(f"  Result: {result}")
            click.echo(f"  FEN: {game.get_fen()}")


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
    import cv2
    from chess_vision.board.detect import detect_board
    from chess_vision.board.warp import compute_homography, warp_board
    from chess_vision.board.squares import extract_squares
    from chess_vision.inference.classify import classify_board, board_to_fen
    from chess_vision.inference.onnx_runtime import ONNXClassifier
    from chess_vision.calibration.profile import load_profile
    from chess_vision.config import ROOT_DIR

    img = cv2.imread(image)
    if img is None:
        click.echo(f"Error: Could not load image {image}")
        return

    corners = detect_board(img)
    if corners is None:
        click.echo("Error: Could not detect board in image.")
        return

    click.echo(f"Board detected with corners at: {corners.tolist()}")

    H = compute_homography(corners)
    warped = warp_board(img, H)
    squares = extract_squares(warped)

    if profile:
        prof = load_profile(profile)
        occ_path = ROOT_DIR / prof.occupancy_model
        piece_path = ROOT_DIR / prof.piece_model
    else:
        occ_path = MODELS_DIR / "occupancy.onnx"
        piece_path = MODELS_DIR / "piece.onnx"

    if occ_path.exists() and piece_path.exists():
        occ_model = ONNXClassifier(occ_path)
        piece_model = ONNXClassifier(piece_path)
        board_state = classify_board(squares, occ_model, piece_model)
        fen = board_to_fen(board_state)
        click.echo(f"FEN: {fen}")
    else:
        click.echo("Warning: Model files not found. Board detected but cannot classify pieces.")
        click.echo("Run 'chess-vision train --stage base' first.")

    if visualize:
        annotated = img.copy()
        for i, corner in enumerate(corners):
            cv2.circle(annotated, tuple(corner.astype(int)), 10, (0, 0, 255), -1)
        cv2.imshow("Detected Board", annotated)
        cv2.imshow("Warped Board", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
