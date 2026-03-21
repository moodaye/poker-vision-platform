# poker-vision-card-classifier

Classifies cropped playing card images into one of 52 card classes (e.g. `KH`, `9S`, `2D`).

Part of the poker bot pipeline:

```
Screen Monitor → Object Detector → Card Snipper → Card Labeller → [training data]
                                                                          ↓
                                                                     train.py
                                                                          ↓
Screen Monitor → Object Detector → Card Snipper → Card Classifier API → label + confidence
```

---

## How it works

### Training (transfer learning)

Rather than training a neural network from scratch — which would require millions of images — this module uses **transfer learning** on **EfficientNet-B0**, a compact CNN pretrained on ImageNet (1.2 million photographs).

The pretrained model already knows how to detect visual primitives: edges, curves, colour regions, corners. These happen to be exactly the features that distinguish playing cards. Training only replaces the final classification layer, adapting those existing features to our 52 card classes.

**Why EfficientNet-B0?**
- Smallest variant in the EfficientNet family (~20MB model file)
- Sufficient for visually distinct, consistently-rendered card images
- Fast inference on CPU — no GPU required
- Strong performance with small datasets (50–200 examples)

**Training approach:**
- Backbone (feature extractor) is **frozen** — pretrained ImageNet weights are preserved
- Only the final linear layer is trained on card images
- Light data augmentation (small rotation, brightness/contrast variation) to improve generalisation
- No horizontal flip — card rank and suit positions are not symmetric

### Inference (API service)

`api.py` loads the saved model at startup and serves predictions over HTTP. The bot (or any downstream service) sends a base64-encoded card crop and receives a label and confidence score.

---

## Files

| File | Purpose |
|---|---|
| `train.py` | Fine-tunes EfficientNet-B0 on labelled snips, saves model to `model/` |
| `api.py` | Flask inference service, POST /classify → label + confidence |
| `pyproject.toml` | Dependencies (torch CPU-only, torchvision, flask, pillow) |
| `model/model.pt` | Saved model weights *(gitignored — regenerate by running train.py)* |
| `model/classes.json` | Index-to-label mapping e.g. `{"0": "2C", "1": "2D", ...}` *(gitignored)* |

---

## Setup

```powershell
cd poker-vision-card-classifier
uv sync
```

Dependencies are resolved from the CPU-only PyTorch index — no CUDA installation required.

---

## Training

Ensure labelled snips exist in `../poker-vision-card-labeller/labels.csv` and the corresponding images are in `../poker-vision-card-snipper/output/`.

Run the audit first to check label quality and distribution:

```powershell
cd ../poker-vision-card-labeller
uv run python audit.py
```

Then train:

```powershell
cd ../poker-vision-card-classifier
uv run python train.py
```

Re-run `train.py` any time you add more labelled examples. The model is overwritten each run.

**Expected output:**
```
Loaded 52 examples across 30 classes (4 skipped)
Classes (30): 2C, 2D, 2S, ...
Training on 52 images for 50 epochs ...
  Epoch   1/50  loss=3.40  acc=7.7%
  Epoch  10/50  loss=2.31  acc=50.0%
  ...
  Epoch  50/50  loss=0.45  acc=88.5%
Saved model   → model/model.pt
Saved classes → model/classes.json
```

Accuracy improves as more labelled snips are added. Target is 100–200 total snips across all 52 classes.

---

## Running the API

```powershell
uv run python api.py
```

Starts a Flask service on port `5001`.

### Endpoints

**`GET /health`**
```json
{ "status": "ok", "classes": 30 }
```

**`POST /classify`**

Request:
```json
{ "image": "<base64 encoded PNG or JPEG>" }
```

Response:
```json
{ "label": "KH", "confidence": 0.97 }
```

### Example (Python)

```python
import base64, requests

with open("card_crop.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:5001/classify",
    json={"image": encoded}
)
print(response.json())  # {"label": "KH", "confidence": 0.97}
```

---

## Design decisions

**Why not use the Roboflow API for card classification (as used in the object detector)?**
The object detector uses Roboflow because training a general object detector from scratch is expensive. The card classifier is a simpler, well-constrained problem (52 classes, consistent visual style) that is straightforward to train locally — and running it locally eliminates API latency and cost in the real-time bot loop.

**Why freeze the backbone during training?**
With fewer than 200 training images, unfreezing the full backbone risks overfitting — the model memorises the training images rather than learning generalisable features. Freezing the backbone and only training the head is the standard approach for small datasets.

**Why no horizontal flip augmentation?**
Card ranks appear in the top-left and bottom-right corners. Flipping horizontally would not produce a valid card image and would confuse the model.

**Why CPU-only PyTorch?**
Training on 50–200 images for 50 epochs takes approximately 2–3 minutes on CPU. The compute overhead of managing a CUDA installation is not justified at this data scale.
