import csv
from pathlib import Path

import cv2

# Snips are read directly from the snipper's output directory.
# Path is relative to this file's location.
_HERE = Path(__file__).parent
IMAGE_DIR = _HERE / ".." / "poker-vision-card-snipper" / "output"
CSV_PATH = _HERE / "labels.csv"

VALID_RANKS = set("AKQJT98765432")
VALID_SUITS = set("SHDC")


def is_valid_label(label: str) -> bool:
    label = label.strip().upper()
    return len(label) == 2 and label[0] in VALID_RANKS and label[1] in VALID_SUITS


def load_labeled_filenames(csv_path: Path) -> set:
    labeled: set[str] = set()
    if not csv_path.exists():
        return labeled

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labeled.add(row["filename"])
    return labeled


def ensure_csv_exists(csv_path: Path) -> None:
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])


def append_label(csv_path: Path, filename: str, label: str) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, label])


def collect_images(image_dir: Path) -> list[tuple[str, Path]]:
    """Walk image_dir recursively and return (relative_key, abs_path) pairs."""
    results = []
    for abs_path in sorted(image_dir.rglob("*")):
        if abs_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            rel_key = str(abs_path.relative_to(image_dir))
            results.append((rel_key, abs_path))
    return results


def main() -> None:
    ensure_csv_exists(CSV_PATH)
    labeled_files = load_labeled_filenames(CSV_PATH)

    all_images = collect_images(IMAGE_DIR)
    remaining = [(key, path) for key, path in all_images if key not in labeled_files]

    if not remaining:
        print("No unlabeled images found.")
        return

    for filename, path in remaining:
        img = cv2.imread(str(path))

        if img is None:
            print(f"Could not read image: {filename}")
            continue

        cv2.imshow("Card Labeler", img)
        cv2.waitKey(500)

        while True:
            raw = (
                input(
                    f"{filename} -> Label (e.g. AS, TD, 7C) | s=skip | d=delete | q=quit: "
                )
                .strip()
                .upper()
            )

            if raw == "Q":
                cv2.destroyAllWindows()
                print("Stopped by user.")
                return

            if raw == "S":
                print(f"Skipped: {filename}")
                break

            if raw == "D":
                path.unlink(missing_ok=True)
                append_label(CSV_PATH, filename, "INVALID")
                print(f"Deleted and marked invalid: {filename}")
                break

            if is_valid_label(raw):
                append_label(CSV_PATH, filename, raw)
                print(f"Saved: {filename} = {raw}")
                break

            print("Invalid label. Use rank+suit, e.g. AS, KH, TD, 7C.")

    cv2.destroyAllWindows()
    print("All done.")


if __name__ == "__main__":
    main()
