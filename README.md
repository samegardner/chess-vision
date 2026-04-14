# chess-vision

Record over-the-board chess games using a USB camera. Outputs PGN files you can import into Chess.com, Lichess, or any chess engine for analysis.

## How it works

Point a camera at your chessboard, click the four corners, and play. The software detects moves in real-time using computer vision and saves the game as a PGN file.

- **YOLO piece detection** (pretrained [ChessCam](https://github.com/Pbatch/CameraChessWeb) model)
- **Temporal smoothing** via exponential moving average
- **Two-move lookahead** for reliable move detection
- **Hand detection** pauses tracking when your hand is over the board
- **Auto-undo** retracts false positive detections
- **Checkmate/game-over detection** via python-chess

## Quick start

```bash
# Clone and install
git clone https://github.com/samegardner/chess-vision.git
cd chess-vision
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Record a game (first time: click the 4 corners of your board)
python scripts/record_game.py --select-corners --white "Your Name" --black "Opponent"

# Next time (uses saved corners)
python scripts/record_game.py --white "Your Name" --black "Opponent"
```

## Corner selection

When prompted, click the corners in this order:
1. **a1** (White's queenside rook corner)
2. **a8** (Black's queenside rook corner)
3. **h8** (Black's kingside rook corner)
4. **h1** (White's kingside rook corner)

This tells the system which side is white, regardless of camera orientation.

## Commands

```bash
# Record a game
python scripts/record_game.py --white "Sam" --black "Magnus"

# Re-select corners (new board/location)
python scripts/record_game.py --select-corners

# List past games
python scripts/list_games.py

# Copy most recent game to clipboard
python scripts/list_games.py --copy 1

# Tune parameters
python scripts/record_game.py --ema 0.3 --greedy-delay 0.5 --interval 0.03
```

## Requirements

- Python 3.11+
- USB camera (tested with Logitech HD 1080p)
- Works on macOS (Apple Silicon) and Linux

## How it works (technical)

1. Camera captures frames at ~20 FPS
2. Frame is cropped to the board region (from saved corners)
3. [LeYOLO](https://github.com/Pbatch/CameraChessWeb) model detects pieces with bounding boxes
4. Detections are mapped to the 64 board squares via perspective-corrected grid
5. An exponential moving average smooths predictions across frames
6. All legal moves (and move pairs) are scored against the smoothed state matrix
7. Moves are accepted via two-move lookahead or greedy fallback (0.4s confirmation)
8. Auto-undo retracts tentative moves that look wrong after 0.5s
9. Hand detection freezes the state when too many pieces are occluded
10. PGN is saved on quit (Ctrl+C or Q in the display window)

## Project structure

```
scripts/record_game.py              # Main recording script
scripts/list_games.py               # Browse recorded games
chess_vision/inference/yolo_detect.py  # YOLO detection + EMA smoothing
chess_vision/game/move_scorer.py       # Move scoring (two-move lookahead)
chess_vision/game/pgn.py               # PGN generation
chess_vision/board/detect.py           # Manual corner selection
models/chesscam_pieces.onnx            # Pretrained piece detection model (4MB)
games/                                 # Saved game history
```

## Credits

- Piece detection model from [CameraChessWeb](https://github.com/Pbatch/CameraChessWeb) by [@Pbatch](https://github.com/Pbatch) (AGPL-3.0)
- Move scoring approach inspired by CameraChessWeb's two-move lookahead system
- Chess logic via [python-chess](https://python-chess.readthedocs.io/)

## License

AGPL-3.0 (due to the ChessCam model dependency)
