# MVP 0 Technical Summary — AI/ML Components

## Screen Monitor
- Uses PIL/ctypes for Win32 window capture and formatting
- Configurable webhook outbound (no local training)

## Object Detector (Roboflow YOLOv8)
- **Problem**: Model underfitted, ~26% `player_me` detection rate early on
- **Solution**: Created auto-annotator to bootstrap training data
  - Converts existing predictions to YOLO format (filtering by confidence threshold)
  - Injects template bounding boxes for missed classes (positions as fractions of poker-table bbox, auto-scales to any resolution)
- **Status**: 156/165 screenshots annotated, 72 with template injections; ready for Roboflow retraining
- **Decision**: Use Roboflow API (not local training) because general object detection requires large labelled datasets

## Card Classifier
- **Model**: EfficientNet-B0 (transfer learning, frozen backbone)
- **Frozen backbone rationale**: <200 training images → risk of overfitting if full model unfrozen
- **Augmentation**: No horizontal flip (card ranks in corners would become invalid)
- **Compute**: CPU-only PyTorch; 50–200 images, 50 epochs ≈ 2–3 min on CPU; CUDA overhead not justified
- **Inference**: Local Flask service (~<100ms) eliminates API latency vs. Roboflow calls
- **I/O**: base64-encoded card crop → label + confidence

## Detection Enricher — OCR Bottleneck & Optimization

### Initial Problem
- **Library**: EasyOCR (deep learning OCR)
- **Latency**: 20–60s cold start + 2–5s per crop = 10–40s OCR total per screenshot (impractical)
- **Root cause**: General-purpose neural model; power overkill for narrow task (digits + "/" on poker HUD)

### Solution
- **Switch to**: pytesseract (Tesseract binary wrapper, C library)
- **Latency**: 0s cold start + ~1–1.2s per crop on Windows (subprocess spawn overhead)
- **Trade-off**: Requires Tesseract binary install + PSM configuration; EasyOCR requires zero config

### Tesseract Configuration
- **PSM modes**: `psm 7` (single-line) + `psm 6` (block)
- **Preprocessing**: normal + strong (greyscale + contrast adjustment)
- **Character whitelist**: numeric profiles use `0123456789/`

### Recent Optimization: Single-Pass OCR
- **Default**: `ocr_max_passes=1` (only PSM 7 + normal preprocessing)
- **Rationale**: Optimizes common case (well-rendered text); reduces per-object from ~2.3s → ~1.0–1.2s
- **Fallback**: Multi-pass path still available via config (`max_passes=0`); retries with PSM 6 + strong preprocessing

### Special Cases
- **"All In" badge fallback**: chip_stack numeric pass returns empty → retry with player_name OCR profile + regex match
  - Normalizes to `"All In"`, applies confidence floor `max(fallback_conf, 0.65)` to pass parser's 0.55 usable threshold

## Spatial Reasoning — Two-Pass System

### Pass 1: `resolve_spatial_relationships`
- Annotates `dealer_button` with nearest player
- Annotates each `chip_stack` with player above it
- Annotates each `bet`/`pot_bet` with nearest player

### Pass 2: `resolve_hero_position`
- Combines fixed-layout geometry + dealer annotation to derive hero seat (BTN/SB/BB)
- Handles both 2-player and 3-player layouts

### Fallback
- When `player_me` not detected by object detector, position defaults to "BTN"
- Full position pipeline (clockwise seat order → BTN/SB/BB) is implemented and tested; detector quality is the limiting factor

## Hand State Parser
- **Confidence gating**: All extraction fields have usable (0.55+) / trusted (0.80+) thresholds
- **Hero position**: Reads `spatial_info["position"]` from enricher
- **Hero turn**: Inferred from turn-halo score + fold button visibility
- **Hero stack**: Prefers chip_stack with owner match to hero_player name; falls back to best confidence
- **Card ordering**: Left-to-right by bbox x-coordinate (consistent 2-card ordering)

## Decision Engine
- Stateless policy: HandState → action + optional amount
- Test harness mode with synthetic hand histories

## Action Executor
- **Recent fix**: Button discovery now recursive (nested children), searches only configured control classes
- **Not hard-coded** to Win32 `Button` class
- **Config**: `button_control_classes` list (now supports custom poker client buttons e.g., `AfxWnd140u`)

## General Infrastructure
- **Service pattern**: Flask HTTP with graceful fallback (missing services return defaults, don't crash)
- **Confidence signals**: Real confidence values throughout (not hardcoded constants)
- **Architecture**: Modular services + state dataclasses, no shared mutable state
- **Test coverage**: 186 tests passing

## Libraries & Stack

| Component | Libraries |
|-----------|-----------|
| Screen capture | PIL, ctypes (Win32) |
| Computer vision | PIL, numpy, OpenCV |
| ML training | PyTorch (CPU-only wheels) |
| ML inference | EfficientNet-B0, Tesseract binary |
| OCR | pytesseract (not EasyOCR) |
| HTTP services | Flask, httpx |
| Object detection | Roboflow YOLOv8 (API-driven) |
| Win32 automation | ctypes, pyautogui, pygetwindow |

## Key Interview Talking Points

1. **Real-World ML Pipeline Challenges**
   - Training data scarcity → Solved with template injection auto-annotator and AI augmentation
   - Model inference latency → Switched from EasyOCR (20–60s cold start) to pytesseract (~1s per crop)
   - Custom environment constraints → Configurable button discovery for non-standard poker client UI

2. **Systems Design Decisions**
   - When to use classical ML (Tesseract) vs deep learning (EasyOCR) → Task specificity & latency requirements
   - Trade-offs between accuracy and latency in real-time systems → Single-pass-first strategy with configurable fallback
   - Configuration-driven architecture for multi-client support → YAML-based control class discovery

3. **Performance Profiling & Optimization**
   - Identified OCR multi-pass as pipeline bottleneck through timing instrumentation
   - Implemented configurable fallback strategy rather than one-size-fits-all solution
   - Maintained backward compatibility (legacy multi-pass still available)

---

> **Issue analysis** (including the six options for Issue #8, recommended approach, and learning notes) has been moved to `ISSUES.md` — see Issue #8 entry.

   