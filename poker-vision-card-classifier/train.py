"""
train.py — Fine-tunes EfficientNet-B0 on labelled card snips.

Run from the poker-vision-card-classifier directory:
    uv run python train.py

What it does:
  - Reads labels.csv from the labeller module
  - Loads the corresponding card snip images from the snipper output
  - Fine-tunes EfficientNet-B0 (ImageNet pretrained) on those images
  - Saves the trained model to model/model.pt
  - Saves the class index mapping to model/classes.json

Re-run any time you add more labelled examples.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
LABELS_CSV = _HERE / ".." / "poker-vision-card-labeller" / "labels.csv"
IMAGE_DIR = _HERE / ".." / "poker-vision-card-snipper" / "output"
MODEL_DIR = _HERE / "model"

VALID_RANKS = set("AKQJT98765432")
VALID_SUITS = set("SHDC")

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224

# ImageNet normalisation — required because we start from ImageNet pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Helpers ───────────────────────────────────────────────────────────────────


def is_valid(label: str) -> bool:
    label = label.strip().upper()
    return len(label) == 2 and label[0] in VALID_RANKS and label[1] in VALID_SUITS


def load_dataset(csv_path: Path, image_dir: Path) -> tuple[list[Path], list[str]]:
    """
    Read labels.csv and return parallel lists of (image_paths, labels).
    Skips invalid labels and missing image files with a warning.
    """
    paths: list[Path] = []
    labels: list[str] = []
    skipped = 0

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().upper()
            if not is_valid(label):
                skipped += 1
                continue
            img_path = image_dir / Path(row["filename"].replace("\\", "/"))
            if not img_path.exists():
                skipped += 1
                continue
            paths.append(img_path)
            labels.append(label)

    logger.info(
        f"Loaded {len(paths)} examples across {len(set(labels))} classes "
        f"({skipped} skipped)"
    )
    return paths, labels


# ── Dataset ───────────────────────────────────────────────────────────────────


class CardDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: list[str],
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        tensor = self.transform(img)
        label_idx = self.class_to_idx[self.labels[idx]]
        return tensor, label_idx


# ── Model ─────────────────────────────────────────────────────────────────────


def build_model(num_classes: int) -> nn.Module:
    """
    Load EfficientNet-B0 with ImageNet weights, freeze the backbone,
    and replace the final linear layer with one sized to our classes.
    Only the new classifier layer will be trained.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze all existing weights — we only want to train the new head
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classification head
    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ── Training ──────────────────────────────────────────────────────────────────


def train() -> None:
    paths, labels = load_dataset(LABELS_CSV, IMAGE_DIR)
    if not paths:
        logger.error("No valid labelled images found. Run the labeller first.")
        return

    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {str(i): c for c, i in class_to_idx.items()}
    num_classes = len(classes)

    logger.info(f"Classes ({num_classes}): {', '.join(classes)}")

    # Small augmentations: slight rotation and brightness/contrast variation.
    # No horizontal flip — card rank/suit positions are NOT symmetric.
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    dataset = CardDataset(paths, labels, class_to_idx, train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cpu")
    model = build_model(num_classes).to(device)

    # Only optimise the new classifier head — backbone is frozen
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"\nTraining on {len(dataset)} images for {EPOCHS} epochs ...\n")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0

        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct += (outputs.argmax(1) == targets).sum().item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(dataset)
            accuracy = correct / len(dataset) * 100
            logger.info(
                f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  acc={accuracy:.1f}%"
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "model.pt"
    classes_path = MODEL_DIR / "classes.json"

    torch.save(
        {"state_dict": model.state_dict(), "num_classes": num_classes},
        model_path,
    )

    with open(classes_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)

    logger.info(f"\nSaved model   → {model_path.resolve()}")
    logger.info(f"Saved classes → {classes_path.resolve()}")


if __name__ == "__main__":
    train()
