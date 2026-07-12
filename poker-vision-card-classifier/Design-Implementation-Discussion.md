# Card Classifier — Design & Implementation Discussion

Running notes on design decisions made during development. Captures the *why*, not just the *what*.

---

## 1. Is transfer learning the right approach?

**Question:** Given that our input is a single cropped screenshot of one playing card with a static, known design — is transfer learning on EfficientNet-B0 the right call? What are the alternatives?

### Alternatives considered

| Approach | Accuracy | Data needed | Fragility | Verdict |
|---|---|---|---|---|
| Template matching (pixel comparison vs 52 reference images) | ~100% on identical renders | 1 reference per card | Breaks on any scaling, compression, or re-skin | Rejected — not reusable, not ML |
| Rule-based corner extraction + OCR | ~99%+ | None | Requires precise crop alignment | Viable but brittle to crop quality |
| Transfer learning — EfficientNet-B0 | 95–99%+ | 50–200 examples | Tolerates minor variation; retrains on design change | **Selected** |
| Tiny CNN from scratch | 90–99%+ | 500–1000+ | Needs more data than we have | Rejected — insufficient data |
| Vision Transformer (ViT) | High | Hundreds–thousands | Heavy; poor with small datasets | Overkill |
| CLIP (zero-shot) | Reasonable | Zero | Less accurate; can't guarantee label format | Interesting but unsuitable for production |

### Decision: EfficientNet-B0 with frozen backbone

**Rationale:**
- We have ~50–200 labelled examples — too few for a scratch-trained CNN, ideal for fine-tuning
- EfficientNet-B0 is the smallest EfficientNet variant (~20MB), sufficient for 52 visually distinct classes
- The pretrained backbone already detects edges, curves, colours, and shapes — exactly the features that distinguish card ranks and suits
- CPU-only inference is fast enough; no GPU required
- Retraining takes 2–3 minutes if the card design changes — no re-engineering needed

**Interview framing:** "Template matching would work for a perfectly static design, but I chose transfer learning because it's robust to minor image variation (compression artifacts, slight perspective), retrainable in minutes if the design changes, and demonstrates real ML engineering skill."

**Notable alternatives for interviews:**
- `MobileNetV3-Small` — slightly smaller footprint, similar accuracy; preferred when model size is the hard constraint
- `ResNet-18` — the classic academic baseline; good to name-drop as the standard benchmark

---

## 2. Preprocessing

**Question:** What preprocessing is required before feeding a card snip into EfficientNet-B0?

### Mandatory steps

| Step | Why |
|---|---|
| **Resize to 224×224** | EfficientNet-B0 expects this input size; spatial feature maps depend on it |
| **Normalize with ImageNet stats** | Pretrained weights assume input in this range; skipping causes erratic training |
| **Convert to 3-channel RGB** | PNG snips may have an alpha channel; model expects exactly 3 channels |

**ImageNet normalization values (must memorise):**
- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`
- Applied after converting image to float tensor in [0, 1]: `(x − mean) / std`

### Recommended

**Aspect-ratio-preserving resize (letterboxing):** Padding the image to square before resizing avoids squashing the portrait card shape. A plain resize distorts rank numerals and suit symbols. Not strictly required (the model can learn to compensate) but cleaner.

### Standard torchvision pipeline

**Inference:**
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Training (adds light augmentation):**
```python
transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### What to avoid

- **Horizontal flip** — card rank/suit corners are not symmetric; flipping creates an invalid card image
- **Any rotation** — cards are always perfectly upright and axis-aligned in this poker client; see § "Augmentation constraints" below
- **Grayscale conversion** — red vs black suit colour is a useful discriminating feature
- **Heavy denoising** — screenshots have minimal noise; no benefit
- **Large affine scale** — small crops (70–90px) lose critical rank pixels when downscaled

---

## 3. Augmentation constraints — lessons learned

**Question:** What augmentation is safe for synthetic data generation on this poker client?

### The constraint: cards are always upright

This poker client's object detector produces **axis-aligned bounding boxes**. Every card crop at inference time is:
- Perfectly vertical — narrow side at the bottom
- No rotation whatsoever — the detector never produces a tilted bbox

This is a hard constraint, not an approximation.

### Why rotation causes regressions

Card rank characters are small — typically 15–25px tall inside an 80px crop. At this scale, a rotation of even 8–10° distorts the character enough to cross rank boundaries:

| Transform | Effect at 80px crop |
|---|---|
| `RandomRotation(±10°)` | T crossbar tilts → visually resembles Q tail or 5 diagonal |
| `RandomRotation(±8°)` | 9 rounded top → resembles Q at slight angle |
| `RandomAffine(scale=0.88)` | 80px → ~70px → rank numerals lose 12% of pixels |

**Observed regression (2026-07-11):** Adding `RandomRotation(±10°)` to `augment.py` and `±8°` to `train.py` caused:
- `9s` → misclassified as `Qs` (fold_btn_open_weak)
- `9s` → misclassified as `5s` (raise_bb_limped_a9s)
- `5d` → misclassified as `6d` (check_bb_unopened)
- Original `Td` → `Qd` bug was NOT fixed (too few genuinely diverse source crops)

All three new errors are rank confusions between visually adjacent characters — exactly the failure mode predicted by rank-boundary blurring.

### Valid augmentation for this domain

| Transform | Valid range | Rationale |
|---|---|---|
| `ColorJitter brightness` | ±20–25% | Screen brightness varies between sessions |
| `ColorJitter contrast` | ±20–25% | Monitor calibration differences |
| `ColorJitter saturation` | ±10% | Minor colour rendering variance |
| `RandomAffine translate` | ±2% | Small bbox position jitter from detector |
| `RandomAffine scale` | 95–105% | Small bbox size jitter from detector |
| `RandomRotation` | **0° — forbidden** | Cards are always upright |
| `RandomHorizontalFlip` | **Forbidden** | Asymmetric card faces |

### When rotation *would* be valid

If this were a physical card recognition system (hand-held camera, live video), rotation augmentation would be essential. The constraint is specific to this screen-capture pipeline where the detector always outputs axis-aligned crops.

---
