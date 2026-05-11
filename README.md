# chess-vision

Record over-the-board chess games using a USB camera. Outputs PGN files you can import into Chess.com, Lichess, or any chess engine for analysis.

## Demo

https://github.com/samegardner/chess-vision/raw/main/assets/demo.mp4

## How it works

Point a camera at your board, click the four corners, and play. Each frame goes through a [LeYOLO](https://github.com/Pbatch/CameraChessWeb) piece detector; detections are mapped to squares via a perspective transform; an EMA smooths predictions across frames; and every legal move (plus two-move pair) is scored against that smoothed state.

Reliability comes from a handful of safety mechanisms:

- **Hand detection** freezes tracking when pieces are occluded.
- **Auto-undo** retracts tentative moves that don't pan out.
- **Overshoot rescue** confirms the current move and pushes the next one when the player plays both before the system can confirm.
- **Checkmate/stalemate/draw** detected via [python-chess](https://python-chess.readthedocs.io/).

## Quick start

```bash
git clone https://github.com/samegardner/chess-vision.git
cd chess-vision
python -m venv .venv
source .venv/bin/activate
pip install -e .

# First run: click a1, a8, h8, h1 in order
python scripts/record_game.py --select-corners --white "Your Name" --black "Opponent"

# Subsequent runs reuse the saved corners
python scripts/record_game.py --white "Your Name" --black "Opponent"
```

Press `Q` in the display window to save the PGN. The most recent game is also copied to your clipboard.

## Useful flags

```bash
--select-corners            # re-click corners (new board or camera angle)
--auto-corners              # auto-detect corners from the YOLO model
--ema 0.3 --greedy-delay 0.5 --interval 0.03   # tune sensitivity vs latency
```

In-game keys: `Q` quit + save, `R` reset, `C` re-detect corners.

## Requirements

- Python 3.11+
- USB camera (tested with Logitech HD 1080p)
- macOS (Apple Silicon) or Linux

## Project structure

```
scripts/record_game.py                 # Main recording loop
chess_vision/inference/yolo_detect.py  # YOLO + EMA + square mapping
chess_vision/game/move_scorer.py       # Two-move lookahead + auto-undo
chess_vision/game/pgn.py               # PGN generation
chess_vision/board/                    # Corner selection (manual + auto)
models/chesscam_pieces.onnx            # Pretrained piece detector (4MB)
```

## Credits

- Piece detection model: [CameraChessWeb](https://github.com/Pbatch/CameraChessWeb) by [@Pbatch](https://github.com/Pbatch) (AGPL-3.0)
- Move scoring approach inspired by CameraChessWeb
- Chess logic via [python-chess](https://python-chess.readthedocs.io/)

## License

AGPL-3.0 (required by the ChessCam model dependency).
