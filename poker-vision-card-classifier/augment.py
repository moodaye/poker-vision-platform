"""
augment.py — Generate synthetic card crop variants for under-represented classes.

For each class with fewer than --target real examples, takes every real crop
and applies spatial + colour transforms to produce synthetic variants.  These
simulate the natural variation seen at inference time (slight bbox extraction
differences, screen brightness changes, minor rotation from screen capture).

Usage
-----
    uv run python augment.py               # default target = 15
    uv run python augment.py --target 20   # higher minimum per class
    uv run python augment.py --dry-run     # report only, no files written
    uv run python augment.py --seed 123    # different random seed

Output
------
    poker-vision-card-snipper/output/synthetic/<LABEL>/aug_001.png …
    poker-vision-card-labeller/labels_synthetic.csv

Re-run any time you add new real examples.  The synthetic folder and CSV are
fully regenerated on each run from the real labels only.

Design notes
------------
- Only real images (from labels.csv) are used as sources.  Synthetic images
  are never re-augmented to avoid cascade artefacts.
- Transforms are chosen to stay within the natural variation range for this
  poker client's static card design:
    * RandomRotation ±10° — detector bbox is never perfectly axis-aligned
    * RandomAffine translate ±5%, scale 88–112% — bbox size/position jitter
    * ColorJitter brightness/contrast ±0.3, saturation ±0.1 — screen rendering
  Horizontal flip is intentionally excluded — card faces are asymmetric.
- The synthetic folder path (relative to snipper output) is recorded in
  labels_synthetic.csv using the same format as labels.csv so train.py loads
  it without modification.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
LABELS_CSV = _HERE / ".." / "poker-vision-card-labeller" / "labels.csv"
SYNTHETIC_CSV = _HERE / ".." / "poker-vision-card-labeller" / "labels_synthetic.csv"
IMAGE_DIR = _HERE / ".." / "poker-vision-card-snipper" / "output"
SYNTHETIC_DIR = IMAGE_DIR / "synthetic"

VALID_RANKS = set("AKQJT98765432")
VALID_SUITS = set("SHDC")

DEFAULT_TARGET = 15


def is_valid(label: str) -> bool:
    label = label.strip().upper()
    return len(label) == 2 and label[0] in VALID_RANKS and label[1] in VALID_SUITS


# ── Augmentation pipeline ─────────────────────────────────────────────────────
#
# Applied to raw PIL images (before letterboxing/resizing).
# fill=0 uses black padding on revealed edges — consistent with LetterboxToSquare.
#
# ROTATION IS INTENTIONALLY ABSENT — cards are always axis-aligned and upright.
# See module docstring for the full reasoning.

AUG_TRANSFORM = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=0
        ),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
    ]
)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_real_examples(csv_path: Path, image_dir: Path) -> dict[str, list[Path]]:
    """Return {label: [image_path, ...]} for all valid, existing real examples."""
    groups: dict[str, list[Path]] = defaultdict(list)
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
            groups[label].append(img_path)

    if skipped:
        logger.debug(f"Skipped {skipped} invalid/missing rows in {csv_path.name}")
    return dict(groups)


# ── Augmentation ──────────────────────────────────────────────────────────────


def generate_synthetic(
    label: str,
    real_paths: list[Path],
    needed: int,
    output_dir: Path,
    dry_run: bool,
) -> list[tuple[str, str]]:
    """
    Generate `needed` synthetic variants from `real_paths`.

    Returns a list of (relative_path_str, label) tuples for the CSV.
    relative_path_str is relative to IMAGE_DIR (poker-vision-card-snipper/output).
    """
    label_dir = output_dir / label
    if not dry_run:
        label_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[tuple[str, str]] = []
    generated = 0
    source_idx = 0

    while generated < needed:
        src_path = real_paths[source_idx % len(real_paths)]
        aug_index = generated + 1
        aug_filename = f"aug_{aug_index:03d}.png"
        aug_rel_path = f"synthetic/{label}/{aug_filename}"

        if not dry_run:
            img = Image.open(src_path).convert("RGB")
            aug_img = AUG_TRANSFORM(img)
            aug_img.save(label_dir / aug_filename)

        csv_rows.append((aug_rel_path, label))
        generated += 1
        source_idx += 1

    return csv_rows


# ── Main ──────────────────────────────────────────────────────────────────────


def run(target: int, dry_run: bool, seed: int) -> None:
    torch.manual_seed(seed)

    # ── Load real examples ─────────────────────────────────────────────────
    if not LABELS_CSV.exists():
        logger.error(f"labels.csv not found at {LABELS_CSV}")
        return

    real_groups = load_real_examples(LABELS_CSV, IMAGE_DIR)
    if not real_groups:
        logger.error("No valid labelled images found in labels.csv")
        return

    # ── Determine which classes need augmentation ──────────────────────────
    needs_aug = {
        label: paths for label, paths in real_groups.items() if len(paths) < target
    }

    total_real = sum(len(v) for v in real_groups.values())
    logger.info(f"Real examples: {total_real} across {len(real_groups)} classes")
    logger.info(f"Target minimum: {target} examples per class")
    logger.info(
        f"Classes below target: {len(needs_aug)} "
        f"({len(real_groups) - len(needs_aug)} already at/above target)"
    )

    if not needs_aug:
        logger.info("All classes already meet the target. Nothing to generate.")
        return

    # ── Clear old synthetic output ─────────────────────────────────────────
    if not dry_run:
        if SYNTHETIC_DIR.exists():
            shutil.rmtree(SYNTHETIC_DIR)
            logger.info(f"Cleared {SYNTHETIC_DIR}")

    # ── Generate ───────────────────────────────────────────────────────────
    all_csv_rows: list[tuple[str, str]] = []
    summary_rows: list[tuple[str, int, int, int]] = []

    for label in sorted(needs_aug):
        real_paths = needs_aug[label]
        real_count = len(real_paths)
        needed = target - real_count

        rows = generate_synthetic(label, real_paths, needed, SYNTHETIC_DIR, dry_run)
        all_csv_rows.extend(rows)
        summary_rows.append((label, real_count, needed, real_count + needed))

    # ── Write labels_synthetic.csv ─────────────────────────────────────────
    if not dry_run and all_csv_rows:
        with open(SYNTHETIC_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            for rel_path, lbl in all_csv_rows:
                writer.writerow([rel_path, lbl])
        logger.info(f"Wrote {len(all_csv_rows)} rows → {SYNTHETIC_CSV}")
    elif dry_run:
        logger.info(f"[DRY RUN] Would write {len(all_csv_rows)} rows → {SYNTHETIC_CSV}")

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print(f"{'Label':<6}  {'Real':>5}  {'+Synth':>7}  {'Total':>6}")
    print("-" * 28)
    for label, real, synth, total in sorted(summary_rows):
        print(f"{label:<6}  {real:>5}  {synth:>7}  {total:>6}")
    print("-" * 28)
    total_synth = sum(s for _, _, s, _ in summary_rows)
    total_after = sum(t for _, _, _, t in summary_rows)
    print(f"{'TOTAL':<6}  {'':>5}  {total_synth:>7}  {total_after:>6} new examples")
    print()

    # Classes already at/above target
    at_target = sorted(
        (lbl, len(paths)) for lbl, paths in real_groups.items() if len(paths) >= target
    )
    if at_target:
        print(f"Classes already at ≥{target} real examples (no augmentation needed):")
        for lbl, cnt in at_target:
            print(f"  {lbl}: {cnt}")
        print()

    if dry_run:
        print("[DRY RUN] No files were written.")
    else:
        print(
            "Done. Re-run train.py to use the synthetic data.\n"
            "train.py auto-loads labels_synthetic.csv when present."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic card crop variants for under-represented classes."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Minimum examples per class (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be generated without writing any files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    run(target=args.target, dry_run=args.dry_run, seed=args.seed)


if __name__ == "__main__":
    main()
