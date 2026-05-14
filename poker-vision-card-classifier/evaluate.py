"""
evaluate.py — Evaluate classifier accuracy against labelled training examples.

Run from the poker-vision-card-classifier directory:
    uv run python evaluate.py
    uv run python evaluate.py --failures-only

Outputs:
  1. Console: per-example true vs predicted, confidence, PASS/FAIL
  2. Console: overall accuracy and per-class failure summary
  3. eval_failures.png: thumbnail grid of every misclassified crop
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models import efficientnet_b0

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
MODEL_PATH = _HERE / "model" / "model.pt"
CLASSES_PATH = _HERE / "model" / "classes.json"
LABELS_CSV = _HERE / ".." / "poker-vision-card-labeller" / "labels.csv"
IMAGE_DIR = _HERE / ".." / "poker-vision-card-snipper" / "output"
FAILURES_OUTPUT = _HERE / "eval_failures.png"

VALID_RANKS = set("AKQJT98765432")
VALID_SUITS = set("SHDC")
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

THUMB = 96  # thumbnail size for the failure grid
COLS = 8  # columns in the failure grid


# ── Model ─────────────────────────────────────────────────────────────────────


class LetterboxToSquare:
    """Pad image to square with black borders, preserving aspect ratio."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(img, ((size - w) // 2, (size - h) // 2))
        return canvas


_transform = transforms.Compose(
    [
        LetterboxToSquare(),
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_model() -> tuple[nn.Module, dict[str, str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train.py first.")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"classes.json not found: {CLASSES_PATH}. Run train.py first."
        )

    with open(CLASSES_PATH) as f:
        idx_to_class: dict[str, str] = json.load(f)

    num_classes = len(idx_to_class)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info(f"Model loaded — {num_classes} classes")
    return model, idx_to_class


def predict(
    model: nn.Module,
    idx_to_class: dict[str, str],
    img: Image.Image,
) -> tuple[str, float]:
    tensor = _transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        conf, idx = probs.max(dim=1)
    return idx_to_class[str(idx.item())], round(conf.item(), 4)


# ── Data ──────────────────────────────────────────────────────────────────────


def is_valid(label: str) -> bool:
    label = label.strip().upper()
    return len(label) == 2 and label[0] in VALID_RANKS and label[1] in VALID_SUITS


def load_examples() -> list[tuple[Path, str]]:
    examples: list[tuple[Path, str]] = []
    with open(LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().upper()
            if not is_valid(label):
                continue
            path = IMAGE_DIR / Path(row["filename"].replace("\\", "/"))
            if path.exists():
                examples.append((path, label))
    return examples


# ── Failure grid ──────────────────────────────────────────────────────────────

_LABEL_H = 14  # pixel height reserved for text below each thumbnail


def save_failure_grid(failures: list[tuple[Path, str, str, float]]) -> None:
    """Save a visual grid of misclassified crops with true→predicted annotation."""
    n = len(failures)
    rows = (n + COLS - 1) // COLS
    cell_h = THUMB + _LABEL_H
    grid = Image.new("RGB", (COLS * THUMB, rows * cell_h), (30, 30, 30))

    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    for i, (path, true, pred, conf) in enumerate(failures):
        img = Image.open(path).convert("RGB")
        img = img.resize((THUMB, THUMB), Image.LANCZOS)
        r, c = divmod(i, COLS)
        x0, y0 = c * THUMB, r * cell_h
        grid.paste(img, (x0, y0))
        draw.text(
            (x0 + 2, y0 + THUMB + 1),
            f"{true}→{pred} {conf:.2f}",
            fill=(255, 80, 80),
            font=font,
        )

    grid.save(str(FAILURES_OUTPUT))
    logger.info(f"Failure grid saved → {FAILURES_OUTPUT} ({n} failures)")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate card classifier accuracy")
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Print only failed predictions (suppress PASS lines)",
    )
    args = parser.parse_args()

    model, idx_to_class = load_model()
    examples = load_examples()
    logger.info(f"Evaluating {len(examples)} labelled examples…")

    correct = 0
    failures: list[tuple[Path, str, str, float]] = []
    per_class: dict[str, list[bool]] = {}

    for path, true_label in examples:
        img = Image.open(path)
        pred_label, conf = predict(model, idx_to_class, img)
        ok = pred_label == true_label

        if ok:
            correct += 1
        else:
            failures.append((path, true_label, pred_label, conf))

        per_class.setdefault(true_label, []).append(ok)

        if ok and args.failures_only:
            continue
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status}  true={true_label:3}  pred={pred_label:3}  conf={conf:.2f}  {path.name}"
        )

    total = len(examples)
    print(f"\nOverall: {correct}/{total} correct ({100 * correct / total:.1f}%)\n")

    failed_classes = {
        cls: results for cls, results in sorted(per_class.items()) if not all(results)
    }
    if failed_classes:
        print("Per-class failures:")
        for cls, results in failed_classes.items():
            n_ok = sum(results)
            print(f"  {cls}: {n_ok}/{len(results)} correct")
        print()

    if failures:
        save_failure_grid(failures)


if __name__ == "__main__":
    main()
