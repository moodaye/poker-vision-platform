# Poker Vision — Card Labeller

Interactive tool for assigning rank+suit labels to snipped card images.

---

## Dependencies

Reads snips directly from `poker-vision-card-snipper/output/` — no manual copying required.

Install all dependencies from the repo root:

```bash
uv sync
```

---

## Usage

```bash
cd poker-vision-card-labeller
python labeller.py
```

Each card image is displayed in a window. Enter a label in the terminal:

| Input | Meaning |
|---|---|
| `AS` | Ace of Spades |
| `KH` | King of Hearts |
| `TD` | Ten of Diamonds |
| `7C` | Seven of Clubs |
| `q` | Quit and save progress |

Valid ranks: `A K Q J T 9 8 7 6 5 4 3 2`
Valid suits: `S H D C`

Labels are saved to `labels.csv` after each card. The session is resumable — already-labelled images are skipped on restart.

---

## Output

`labels.csv` — columns: `filename`, `label`

The `filename` key is the path relative to the snipper output directory, e.g.:
```
capture_20260219_174930_717002\card_00.png, AS
```
