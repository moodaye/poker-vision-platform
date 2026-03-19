"""
audit.py - Label quality audit for the card labeller.

Run with:
    uv run python audit.py

Outputs:
  1. Console: count of labelled snips per card, sorted by suit/rank.
  2. audit_grid.png: visual grid of all snips grouped by label, for
     spot-checking that each card is in the correct group.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent
IMAGE_DIR = _HERE / ".." / "poker-vision-card-snipper" / "output"
CSV_PATH = _HERE / "labels.csv"
GRID_OUTPUT = _HERE / "audit_grid.png"

VALID_RANKS = set("AKQJT98765432")
VALID_SUITS = set("SHDC")


def is_valid(label: str) -> bool:
    label = label.strip().upper()
    return len(label) == 2 and label[0] in VALID_RANKS and label[1] in VALID_SUITS


# ── Data loading ─────────────────────────────────────────────────────────────


def load_labels(
    csv_path: Path,
) -> tuple[dict[str, list[Path]], list[str], list[tuple[str, str]]]:
    """
    Read labels.csv and group image paths by label.

    Returns:
        groups:  {label: [abs_image_path, ...]} for valid, found images
        missing: list of path strings where the image file was not found
        invalid: list of (filename, label) for non-valid labels (e.g. INVALID)
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            label = row["label"]

            if not is_valid(label):
                invalid.append((filename, label))
                continue

            # Normalise Windows backslashes so Path works cross-platform
            img_path = IMAGE_DIR / Path(filename.replace("\\", "/"))
            if not img_path.exists():
                missing.append(str(img_path))
                continue

            groups[label].append(img_path)

    return dict(groups), missing, invalid


def _sort_key(label: str) -> tuple:
    """Sort labels by suit (SHDC) then rank (AKQJT98765432)."""
    suit_order = {s: i for i, s in enumerate("SHDC")}
    rank_order = {r: i for i, r in enumerate("AKQJT98765432")}
    return (suit_order[label[1]], rank_order[label[0]])


# ── Console report ────────────────────────────────────────────────────────────


def print_report(
    groups: dict[str, list[Path]],
    missing: list[str],
    invalid: list[tuple[str, str]],
) -> None:
    present = {label: len(paths) for label, paths in groups.items()}
    suit_names = {"S": "Spades", "H": "Hearts", "D": "Diamonds", "C": "Clubs"}

    print("\n── Label distribution ──────────────────────────────────────────")
    for suit in "SHDC":
        print(f"\n  {suit_names[suit]}:")
        for rank in "AKQJT98765432":
            label = f"{rank}{suit}"
            count = present.get(label, 0)
            bar = "█" * count
            flag = "  ← none yet" if count == 0 else ""
            print(f"    {label}: {count:3d}  {bar}{flag}")

    total = sum(present.values())
    print(f"\n  Total valid snips : {total}")
    print(f"  Unique cards seen : {len(present)} / 52")

    if invalid:
        print(f"\n── Skipped (invalid label): {len(invalid)} ──────────────────────")
        for fn, lbl in invalid:
            print(f"    {lbl:10s}  {fn}")

    if missing:
        print(f"\n── Skipped (image not found): {len(missing)} ─────────────────────")
        for p in missing:
            print(f"    {p}")


# ── Visual grid ───────────────────────────────────────────────────────────────

THUMB_W = 64
THUMB_H = 90
LABEL_W = 36  # pixels reserved for the label text on the left
PAD = 4
BG_COLOUR = (40, 40, 40)
TEXT_COLOUR = (220, 220, 220)
DIVIDER_COLOUR = (80, 80, 80)


def make_grid(groups: dict[str, list[Path]]) -> np.ndarray | None:
    """
    Build a NumPy image arranged as:
        [label]  [snip0] [snip1] [snip2] ...
    One row per label, sorted by suit then rank.
    """
    if not groups:
        return None

    sorted_labels = sorted(groups.keys(), key=_sort_key)
    max_cols = max(len(paths) for paths in groups.values())

    row_h = THUMB_H + PAD * 2
    col_w = THUMB_W + PAD
    total_w = LABEL_W + max_cols * col_w + PAD
    total_h = len(sorted_labels) * row_h + PAD

    canvas = np.full((total_h, total_w, 3), BG_COLOUR, dtype=np.uint8)

    for row_idx, label in enumerate(sorted_labels):
        y = PAD + row_idx * row_h

        # Horizontal divider between rows
        if row_idx > 0:
            canvas[y - PAD // 2, :] = DIVIDER_COLOUR

        # Label text
        cv2.putText(
            canvas,
            label,
            (2, y + THUMB_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            TEXT_COLOUR,
            1,
            cv2.LINE_AA,
        )

        # Thumbnails
        for col_idx, img_path in enumerate(groups[label]):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            thumb = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            x = LABEL_W + col_idx * col_w
            canvas[y : y + THUMB_H, x : x + THUMB_W] = thumb

    return canvas


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: labels.csv not found at {CSV_PATH}")
        sys.exit(1)

    groups, missing, invalid = load_labels(CSV_PATH)
    print_report(groups, missing, invalid)

    grid = make_grid(groups)
    if grid is not None:
        cv2.imwrite(str(GRID_OUTPUT), grid)
        print(f"\n── Visual grid saved → {GRID_OUTPUT.resolve()}\n")
    else:
        print("\nNo valid images found — grid not generated.\n")


if __name__ == "__main__":
    main()
