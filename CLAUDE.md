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
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

- `chess_vision/board/` - Board detection, perspective warp, square extraction
- `chess_vision/models/` - CNN model definitions and ONNX export
- `chess_vision/training/` - Dataset classes, augmentation, training loops
- `chess_vision/inference/` - Camera, ONNX runtime, full classification pipeline
- `chess_vision/game/` - Game state, move detection, legal move resolution, PGN output
- `chess_vision/calibration/` - Guided capture, auto-labeling, profile management
