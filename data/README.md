# Training Data

This directory holds training datasets (gitignored due to size).

## Downloading

```bash
python scripts/download_data.py
```

This downloads:
- `chesscog/` - ~5,000 synthetic board images rendered in Blender (chesscog project)
- `chessred/` - 10,800 real smartphone photos from ChessReD dataset

## Manual placement

You can also place raw test photos in `raw/` for debugging board detection.
