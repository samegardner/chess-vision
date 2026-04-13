"""Data augmentation pipelines for training."""

from torchvision import transforms

from chess_vision.config import INPUT_SIZE


def get_train_transforms() -> transforms.Compose:
    """Standard training augmentations."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_calibration_transforms() -> transforms.Compose:
    """Aggressive augmentations for fine-tuning from small calibration set."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Eval/inference transforms: resize + normalize only."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
