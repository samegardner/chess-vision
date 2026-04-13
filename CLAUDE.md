# chess-vision

CNN-powered chess board recognition from a USB camera. Records OTB games as PGN files.

## Architecture

Two-stage CNN pipeline:
1. **Board detection**: Hough lines + RANSAC corner detection, homography warp to 800x800
2. **Occupancy CNN**: ResNet-18 binary classifier (empty vs occupied)
3. **Piece CNN**: ResNet-18 12-class classifier (6 piece types x 2 colors)
4. **Move detection**: Square-level frame differencing with 5-frame stability window
5. **Game logic**: python-chess for legal move validation, FEN/PGN generation

## Key commands

```bash
chess-vision calibrate --name "set-name"     # Calibrate for a chess set
chess-vision record --profile "set-name"     # Record a game
chess-vision train --stage base              # Train base models
chess-vision detect --image photo.jpg        # One-shot board detection
chess-vision list-profiles                   # List calibration profiles
```

## Development

```bash
cd ~/Personal/Projects/chess-vision
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Virtual environment is at `.venv/` (Python 3.13). Always activate before running commands.

## Training pipeline

```bash
# 1. Download and process training data (chesscog synthetic dataset from OSF)
python scripts/download_data.py --dataset chesscog

# 2. Train base models (uses MPS on Apple Silicon)
chess-vision train --stage base --epochs 20

# 3. Models saved to models/occupancy.onnx and models/piece.onnx
```

Training data goes to `data/chesscog/` (raw) and `data/processed/` (per-square crops). Both are gitignored.

## Project layout

- `chess_vision/board/` - Board detection, perspective warp, square extraction
- `chess_vision/models/` - CNN model definitions (ResNet-18), ONNX export. **Single source of truth for PIECE_CLASSES and PIECE_TO_FEN** (in `piece.py`)
- `chess_vision/training/` - Dataset classes (ImageFolder), augmentation, training loop (Adam + cosine LR + early stopping)
- `chess_vision/inference/` - Camera wrapper, ONNX Runtime inference, full board classification pipeline
- `chess_vision/game/` - Game state, move detection (frame diff + stability window), legal move resolution (handles castling, en passant, promotion), PGN export
- `chess_vision/calibration/` - Guided photo capture, auto-labeling from starting position, per-set fine-tuning, profile management (YAML)
- `scripts/` - Dataset download/processing, debug visualization
- `config/default.yaml` - All configurable parameters (note: config loading exists but module constants are currently hardcoded, not populated from YAML)

## Key design decisions

- **Two-stage classification** (occupancy then piece type) is proven more accurate than single 13-class. Occupancy filters 60% of squares so piece model only runs on ~16-32 squares.
- **ONNX Runtime** for inference (2-5x faster than PyTorch on CPU, no GPU required at inference time)
- **50% contextual padding** on square crops captures full piece silhouette at angled camera views
- **Legal move validation** as error correction: python-chess constrains CNN output to only legal moves, dramatically reducing effective error rate
- **Per-set calibration**: 2 photos of starting position (128 labeled squares) fine-tune both models via transfer learning
- **Board orientation**: auto-detected from starting position, `remap_board_state()` handles 180-degree flip when playing as black

## Known limitations

- Board detection uses traditional CV (Hough lines), not ML. Works well on clean boards but may struggle with cluttered backgrounds or poor lighting.
- Cached homography during recording means a shifted camera produces wrong classifications with no warning. Periodic re-detection is planned but not yet implemented.
- Config YAML (`config/default.yaml`) is defined but module constants are hardcoded. `load_config()` exists but doesn't update the constants.

## Testing

```bash
pytest                    # Run all tests (55 tests)
pytest tests/test_rules.py  # Run specific test file
pytest -v --tb=short      # Verbose with short tracebacks
```

Tests use synthetic checkerboard images for board detection. No camera or GPU required.
