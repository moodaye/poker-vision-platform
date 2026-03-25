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
- **Large rotations** — cards are always upright in screenshots
- **Grayscale conversion** — red vs black suit colour is a useful discriminating feature
- **Heavy denoising** — screenshots have minimal noise; no benefit

---
