# InfraDefect — Technical Decisions Log

---

## D-001 — Model: SegFormer-B2

**Status:** Accepted

**Context:** Need a semantic segmentation model to detect thin cracks, spalling regions, corrosion staining, efflorescence, and exposed rebar at pixel level. Must run on a single consumer GPU (RTX 3080 or similar) with reasonable inference latency.

**Decision:** SegFormer-B2 with ImageNet-pretrained MiT-B2 encoder via HuggingFace Transformers.

**Alternatives rejected:**
- DeepLabV3+ (ResNet-50) — 72% more parameters for lower mIoU; dilated convolutions less effective for long-range crack topology than self-attention
- Mask2Former (Swin-T) — highest accuracy but ~80ms inference and 6GB+ VRAM for batch processing; overkill for a 5-class problem
- U-Net (EfficientNet-B4) — fast and excellent on blob-shaped regions; rejected because local skip connections miss continuous crack paths that require global context

**Trade-offs accepted:**
- MLP decoder produces slightly softer mask boundaries than U-Net skip connections; mitigated by boundary loss term
- Transformer attention less interpretable than GradCAM on CNNs

**Revisit if:** crack IoU remains below 0.35 after full training; evaluate Mask2Former or ensemble.

---

## D-002 — Training: Two-Stage Fine-Tuning

**Status:** Accepted

**Context:** The pretrained `nvidia/mit-b2` checkpoint has an ImageNet classification head. The segmentation decode head is randomly initialised. Training all layers simultaneously from epoch 1 risks catastrophic forgetting of encoder features before the decoder has stabilised.

**Decision:** Stage 1 freezes the encoder for 5 epochs (decoder warmup), Stage 2 fine-tunes all layers with layer-wise learning rates (encoder 6e-5, decoder 6e-4).

**Alternatives rejected:**
- Single-stage full fine-tuning from epoch 1 — early unstable gradients from the random decoder head damage pretrained encoder representations
- Frozen encoder throughout — limits the model's ability to adapt low-level features (texture, edge responses) to the defect domain

**Trade-offs accepted:**
- Two-stage adds complexity to the training loop; adds one extra checkpoint (stage1_final.pth)
- Stage 1 epoch count (5) is a heuristic; may need tuning for different dataset sizes

---

## D-003 — Loss Function: BCE + Dice + Boundary

**Status:** Accepted

**Context:** Defect classes are multi-label (an image can have multiple classes simultaneously) and highly imbalanced at the pixel level — background dominates, cracks are very sparse.

**Decision:** Composite loss: `L = 0.4 * BCE + 0.4 * Dice + 0.2 * Boundary`

BCE uses class-frequency-derived `pos_weight` to balance positive pixel gradients. Dice directly optimises overlap. Boundary loss penalises edge errors.

**Alternatives rejected:**
- Cross-entropy alone — heavily biased toward background, ignores class imbalance
- Focal loss — effective but requires careful gamma tuning; Dice achieves similar effect with less sensitivity
- Lovász loss — directly optimises IoU but known training instability in early epochs

**Trade-offs accepted:**
- Boundary loss adds ~15% training time overhead per batch (edge map computation)
- Three-component loss makes learning rate sensitivity harder to reason about

**Revisit if:** minority class F1 below 0.50 after full training; consider adding class-weighted CE or Lovász as a fourth term.

---

## D-004 — Classes: 5 Defect Types

**Status:** Accepted

**Context:** Original CODEBRIM dataset has 6 classes. The Roboflow export uses the label format `"{index} {ClassName}"` (e.g. `"2 Crack"`, `"1 CorrosionStain"`). Efflorescence and corrosion were initially merged into one class — incorrect because they are visually and structurally distinct defects.

**Decision:** 5 classes: efflorescence (0), corrosion (1), crack (2), spalling (3), exposed_bars (4). Each class has a fixed colour and Otsu polarity (bright vs dark relative to background).

**Key correction made during development:** Initial schema had 4 classes with efflorescence merged into corrosion. This caused mIoU=0.0000 during training because the label `"0 Efflorescence"` (index 4 in the export) was out of bounds for the 4-column label matrix. Corrected to 5 classes.

**Trade-offs accepted:**
- 5-class model has marginally more parameters in the decode head classifier layer than 4-class
- Required updating CLASS_MAP, INVERT_OTSU, pos_weight shape, and NUM_CLASSES across all files

---

## D-005 — Mask Generation: Otsu Thresholding from Bounding Boxes

**Status:** Accepted

**Context:** The Roboflow dataset provides bounding box annotations, not pixel-level masks. SegFormer requires pixel masks for training.

**Decision:** For each bounding box, crop the image region and apply Otsu thresholding to separate the defect from background. Bright defects (efflorescence, corrosion) use standard Otsu; dark defects (crack, spalling, exposed_bars) use inverted Otsu.

**Alternatives rejected:**
- Use bounding boxes directly as rectangular masks — introduces too much background into the mask, confusing the model
- Skip training, use zero-shot model — no suitable zero-shot model exists for this specific defect taxonomy
- Manual pixel annotation — accurate but expensive; not feasible for 1051 images without a labelling team

**Trade-offs accepted:**
- Otsu-derived masks are approximate; complex textures or mixed-class regions produce noisy masks
- Crack masks from bounding boxes are particularly noisy because cracks are thin relative to their bounding box area
- Model learns from imperfect masks; final accuracy is bounded by mask quality

**Revisit if:** IoU plateaus and does not improve past 0.45 mIoU; invest in pixel-level annotation for at least the crack and efflorescence classes.

---

## D-006 — Input Resolution: 512x512

**Status:** Accepted

**Context:** Input images range from 640x480 (phone) to 4000x3000 (DSLR). Need consistent batch sizing for training and inference.

**Decision:** All images letterbox-padded and resized to 512x512. Aspect ratio is preserved; shorter dimension is padded with black.

**Alternatives rejected:**
- 256x256 — hairline cracks become sub-pixel, model cannot learn fine texture
- 1024x1024 — batch size constrained to 1-2 on RTX 3080, training instability, 4x slower inference
- Multi-scale inference (256/512/1024 merge) — 3x inference cost with diminishing accuracy returns

**Trade-offs accepted:**
- Very high-resolution images lose fine detail; sub-millimetre cracks in 4K images may not be detected
- Black padding introduces artificial borders; model learns to ignore these through augmentation

**Revisit if:** crack IoU below 0.35 after full training; run 1024x1024 ablation to check if resolution is the bottleneck.

---

## D-007 — API Framework: FastAPI

**Status:** Accepted

**Context:** Need a Python REST API server with file upload, async support, and auto-generated documentation.

**Decision:** FastAPI with Uvicorn ASGI server.

**Alternatives rejected:**
- Django REST Framework — ORM, admin panel, session management are unnecessary overhead for a stateless inference API
- Flask — synchronous by default, requires extensions for async, no automatic OpenAPI docs
- Triton Inference Server — production-grade but steep learning curve, overkill for single-model serving

**Trade-offs accepted:**
- No built-in job queue; uploads processed synchronously one at a time
- Younger ecosystem than Django/Flask; fewer community resources for edge cases

**Revisit if:** concurrent users exceed ~10 requests/min or inference needs to be decoupled from the HTTP response cycle; add Celery + Redis.

---

## D-008 — Frontend: React (Create React App)

**Status:** Accepted

**Context:** Need a polished UI for image upload, overlay rendering, and per-class statistics display. Requirement specified React + JavaScript.

**Decision:** React SPA using Create React App, with custom CSS-in-JS styling. No UI component library — all components written from scratch for full visual control.

**Alternatives rejected:**
- Gradio — cannot meet React requirement; limited customisation for multi-class colour overlay rendering
- Streamlit — same reason; poor canvas control
- Next.js — SSR unnecessary for a pure client-side inference UI; CRA simpler to deploy

**Trade-offs accepted:**
- Requires separate process (port 3000) vs embedding in FastAPI; handled with CORS middleware and CRA proxy setting
- No SSR means slower initial load on cold start; acceptable for internal tooling

---

## D-009 — Checkpoint Strategy: Best + Last

**Status:** Accepted

**Context:** Training metrics can diverge — the last epoch may have lower val loss but not the best mIoU.

**Decision:** Save two checkpoints per training run:
- `best_model.pth` — highest val mIoU achieved (used for inference)
- `last.pth` — final epoch (used for resuming training)
- `stage1_final.pth` — end of Stage 1 (decoder warmup complete)

**Trade-offs accepted:**
- ~300MB total disk usage for three checkpoints
- Requires tracking best_miou across epochs and conditional save logic

---

## D-010 — Stratified Split: skmultilearn

**Status:** Accepted

**Context:** Multi-label stratification is not supported by scikit-learn's `StratifiedKFold` (which handles single-label only). Defect classes are imbalanced and co-occur (e.g. crack + corrosion in the same image).

**Decision:** Use `skmultilearn.model_selection.iterative_train_test_split` for 70/15/10 stratified split. Falls back to random split with a warning if skmultilearn is not installed.

**Trade-offs accepted:**
- Adds `scikit-multilearn` dependency
- Iterative stratification is a greedy heuristic; not guaranteed optimal but consistently better than random for multi-label data