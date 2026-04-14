# chess-vision

Records OTB chess games using a USB camera. Outputs PGN files.

## How it works

1. Camera captures frames of the board
2. ChessCam's pretrained LeYOLO model detects pieces with bounding boxes
3. Detections mapped to squares via perspective transform
4. EMA smoothing stabilizes predictions across frames
5. Legal moves scored against the state matrix (ChessCam's approach)
6. Two-move lookahead + greedy fallback with 1s confirmation
7. python-chess validates and tracks game state
8. PGN exported on quit

## Usage

```bash
# Record a game (select corners first time)
python scripts/record_game.py --select-corners --white "Sam" --black "Opponent"

# Use saved corners
python scripts/record_game.py --white "Sam" --black "Opponent"

# Adjust sensitivity
python scripts/record_game.py --ema 0.3 --greedy-delay 0.8
```

## Project layout (active files only)

```
scripts/record_game.py              # Main recording script
chess_vision/inference/yolo_detect.py  # YOLO detection + EMA + square mapping
chess_vision/game/move_scorer.py       # Move scoring (two-move lookahead)
chess_vision/game/pgn.py               # PGN generation
chess_vision/board/detect.py           # Manual corner selection
chess_vision/board/warp.py             # Corner ordering + homography
models/chesscam_pieces.onnx            # Pretrained LeYOLO model (4MB)
corners.json                           # Saved board corners
```

## Key technical details

- Model: ChessCam's LeYOLO (480x288 input, 12 piece classes, float16)
- Preprocessing: letterbox resize with gray padding (114), matching ChessCam
- Anchor point: bottom of box minus width/3 (piece base, not box center)
- Square centers: computed via inverse homography (handles perspective)
- Out-of-board filtering: point-in-quad test discards off-board detections
- EMA decay: 0.4 (state reacts to changes in ~2 frames)
- Greedy delay: 1.0s (move must be top candidate for 1 second)
- Undo mechanism: if a move looks wrong after 5 frames, automatically reverts

## Development

```bash
source .venv/bin/activate
pytest  # Run tests
```
