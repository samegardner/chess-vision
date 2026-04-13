# Trained Models

This directory holds trained model files (gitignored due to size).

## Expected files after training

- `occupancy.onnx` - Base occupancy classifier (empty vs occupied)
- `piece.onnx` - Base piece classifier (12 piece types)
- `checkpoints/` - Training checkpoint .pt files

## Calibration models

Per-set fine-tuned models are stored alongside their calibration profiles in `profiles/`.
