# InfraDefect — System Architecture

## 1. Overview

InfraDefect is a full-stack pipeline for pixel-level detection of structural defects in concrete and infrastructure inspection images. It accepts JPEG/PNG uploads, runs multi-label semantic segmentation inference, and returns colour-coded overlay images with per-class coverage statistics and severity ratings.

**Five defect classes:**

| Index | Class | Otsu Polarity | Notes |
|---|---|---|---|
| 0 | efflorescence | Bright | White mineral deposits — brighter than background |
| 1 | corrosion | Bright | Orange/brown rust staining — brighter than grey concrete |
| 2 | crack | Dark | Thin dark lines — darker than background |
| 3 | spalling | Dark | Exposed aggregate patches — darker than background |
| 4 | exposed_bars | Dark | Metal rebar — darker than background |

---

## 2. System Diagram

```
┌──────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                        │
│                                                          │
│  React SPA (port 3000)                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  DropZone   │  │ ResultPanel  │  │     Header     │  │
│  │  (upload)   │  │ (overlay +   │  │ (status + API  │  │
│  │             │  │  class stats)│  │  health poll)  │  │
│  └──────┬──────┘  └──────▲───────┘  └────────────────┘  │
│         │                │                               │
└─────────┼────────────────┼───────────────────────────────┘
          │ POST /api/predict  JSON + base64 overlay
          ▼                │
┌──────────────────────────────────────────────────────────┐
│                      API LAYER                           │
│                                                          │
│  FastAPI + Uvicorn (port 8000)                           │
│                                                          │
│  POST /api/predict    GET /api/health                    │
│  GET  /api/classes    GET /api/model-info                │
│  GET  /results/*      (static overlay/mask files)        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │                  ModelEngine                       │  │
│  │  preprocess → forward → postprocess → visualise   │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                     MODEL LAYER                          │
│                                                          │
│  SegFormerForSemanticSegmentation (nvidia/mit-b2)        │
│  - Encoder: Mix Transformer B2 (ImageNet pretrained)     │
│  - Decoder: All-MLP head (5 output channels)             │
│  - Weights: checkpoints/best_model.pth                   │
│  - Input:   512 x 512 RGB                                │
│  - Output:  5 x H x W sigmoid probability maps          │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    STORAGE LAYER                         │
│                                                          │
│  uploads/       Temporary uploaded images                │
│  results/       Overlay + mask PNGs (served as static)   │
│  checkpoints/   Model weights + training logs            │
│  roboflow_data/ Annotated training images (VOC-XML)      │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Model

### Architecture: SegFormer-B2

SegFormer (Xie et al., NeurIPS 2021) pairs a hierarchical Mix Transformer (MiT) encoder with a lightweight all-MLP decoder. The B2 variant was selected as the best balance of accuracy, speed, and memory for this task.

| Model | Params | Inference (GPU) | Why considered |
|---|---|---|---|
| SegFormer-B2 (chosen) | 25M | ~30ms | Best efficiency/accuracy trade-off |
| DeepLabV3+ ResNet-50 | 43M | ~45ms | Strong baseline but heavier, conv-only |
| Mask2Former Swin-T | 47M | ~80ms | Higher accuracy, too slow and memory-heavy |
| U-Net EfficientNet-B4 | 19M | ~22ms | Fast but weak on long thin cracks |

SegFormer was chosen over the alternatives because its self-attention captures long-range spatial context, which is critical for detecting continuous crack paths that CNN receptive fields can miss. U-Net's local skip connections are well-suited to blob-shaped defects but underperform on thin linear structures.

### Training: Two-Stage Fine-Tuning

The pretrained checkpoint from HuggingFace (`nvidia/mit-b2`) was trained for ImageNet classification. The classification head is discarded and a new segmentation decode head is randomly initialised.

**Stage 1 — Decoder warmup (5 epochs)**
- Encoder frozen, only decode_head trained
- Learning rate: 6e-4 (AdamW)
- Rationale: let the new randomly-initialised head stabilise before touching the pretrained encoder

**Stage 2 — Full fine-tuning (45 epochs)**
- All layers unfrozen with layer-wise learning rates
- Encoder LR: 6e-5 (10x smaller — preserve ImageNet features)
- Decoder LR: 6e-4
- Schedule: linear warmup over first 10% of steps, cosine decay to zero
- Gradient clipping: max norm 1.0
- Mixed precision (AMP)
- Early stopping: patience 10 epochs on val mIoU

### Loss Function

```
L = 0.4 * L_BCE + 0.4 * L_Dice + 0.2 * L_Boundary
```

- **BCE with pos_weight**: class-frequency-weighted binary cross-entropy handles per-pixel multi-label prediction and class imbalance
- **Dice loss**: directly optimises overlap metric, complementary to BCE on sparse classes
- **Boundary loss**: penalises mask edge errors, improving sharpness on thin cracks

### Evaluation Metrics

- **IoU (Intersection over Union)** per class: primary metric, used for checkpoint selection
- **F1 score** per class: harmonic mean of precision and recall
- **mIoU**: mean IoU across all 5 classes — used for early stopping and best model selection

Current baseline results (Stage 2, Epoch 1, 1051 images):

| Class | IoU | F1 |
|---|---|---|
| efflorescence | 0.3473 | 0.5155 |
| corrosion | 0.4043 | 0.5758 |
| crack | 0.1410 | 0.2471 |
| spalling | 0.3278 | 0.4937 |
| exposed_bars | 0.4163 | 0.5879 |
| **mean** | **0.3273** | **0.4840** |

Crack scores lower than other classes because thin lines (1-3px wide) are heavily penalised by IoU — even a 1px offset produces near-zero overlap. This improves substantially with more Stage 2 epochs.

---

## 4. Data Pipeline

### Dataset

1051 annotated images in VOC-XML format exported from Roboflow. Each image has a paired `.xml` file with bounding box annotations per defect instance.

Label names follow the format `"{index} {ClassName}"` (e.g. `"2 Crack"`, `"1 CorrosionStain"`). The dataset class normalises these to lowercase and maps them to integer indices via `CLASS_MAP`.

### Multi-Label Stratified Split

Standard random splits risk having rare classes missing entirely from validation. The pipeline uses `skmultilearn.model_selection.iterative_train_test_split` to produce a stratified 70/15/10 split that preserves the class co-occurrence distribution across train, val, and test sets.

Split results on the current dataset:

| Class | Train | Val |
|---|---|---|
| efflorescence | 30.2% | 29.9% |
| corrosion | 52.9% | 54.1% |
| crack | 60.2% | 59.9% |
| spalling | 52.8% | 52.9% |
| exposed_bars | 41.6% | 41.4% |

### Mask Generation from Bounding Boxes

The dataset provides bounding boxes, not pixel masks. Segmentation masks are synthesised per bounding box using Otsu thresholding on the cropped region:

- Bright defects (efflorescence, corrosion): standard Otsu — pixels brighter than threshold → defect
- Dark defects (crack, spalling, exposed_bars): inverted Otsu — pixels darker than threshold → defect

This is an approximation. Accuracy improves with pixel-level annotation but bounding-box-derived masks are sufficient for training with augmentation.

### Augmentation Pipeline (training only)

```
HorizontalFlip (p=0.5)
VerticalFlip (p=0.3)
RandomRotate90 (p=0.3)
ShiftScaleRotate (p=0.4)
ElasticTransform (p=0.25)
GridDistortion (p=0.2)
RandomBrightnessContrast (p=0.4)
HueSaturationValue (p=0.3)
GaussianBlur (p=0.2)
GaussNoise (p=0.2)
CoarseDropout (p=0.2)
```

Normalisation: ImageNet mean/std applied after augmentation.

### Class Weights

`pos_weight` for BCE loss is computed from training set class frequencies:

```
pos_weight[c] = (1 - freq[c]) / freq[c]   clipped to [1.0, 10.0]
```

This balances the gradient contribution of positive pixels against the heavily dominant background.

---

## 5. Backend API

### `backend/app.py`

FastAPI application. Loaded once at startup — the model is kept in GPU memory and reused across requests.

**Request flow:**

```
POST /api/predict
  1. Validate file type (JPEG/PNG only)
  2. Decode image with PIL
  3. ModelEngine.predict(image)
     a. Resize to 512x512 (letterbox padding)
     b. Normalise with ImageNet stats
     c. model.forward() → [1, 5, 128, 128] logits
     d. Bilinear upsample → [1, 5, H, W]
     e. Sigmoid → probability maps
     f. Threshold at 0.45
     g. Morphological cleanup (remove blobs < 50px)
     h. Per-class statistics + severity
     i. Colour overlay generation
  4. Save overlay + mask PNGs to results/
  5. Return JSON with base64 overlay + stats
```

**CORS**: configured to allow all origins (`*`) for local development. Restrict to specific origins in production.

**Static files**: result images served at `/results/<filename>` via FastAPI `StaticFiles` mount. Frontend uses the `overlay_b64` field directly to avoid CORS issues with static file URLs.

---

## 6. Frontend

React SPA using `react-dropzone` for upload and `axios` for API calls. No UI framework — all styling is custom CSS-in-JS with a dark industrial aesthetic matching the domain.

**Components:**

- `Header` — logo, 5-class colour legend, live API status indicator (polls `/api/health` every 15s), engine mode badge (MODEL vs DEMO)
- `DropZone` — drag-and-drop or click-to-browse upload zone, image preview, loading overlay during inference
- `ResultPanel` — overlay/mask toggle viewer, animated per-class stat cards with progress bars and severity colour coding, inference summary footer

**API hooks (`useApi.js`):**

- `useHealth()` — polls `/api/health` on mount and every 15 seconds
- `usePredict()` — sends multipart form POST to `/api/predict`, manages loading/error/result state

**Proxy**: `package.json` sets `"proxy": "http://localhost:8000"` so all `/api/*` requests are forwarded to the backend without CORS issues in development.

---

## 7. Post-Processing

After sigmoid thresholding:

1. **Morphological cleanup** — `cv2.connectedComponentsWithStats` removes isolated blobs smaller than 50px². Eliminates salt-and-pepper noise from the threshold.
2. **Colour overlay** — each defect class is painted at its class colour with alpha blending (α=0.45) over the original image.
3. **Severity classification** — each class's pixel coverage as a percentage of total image area is mapped to a severity level.

---

## 8. Known Limitations

**Bounding box masks** — training masks are approximated from bounding boxes using Otsu thresholding. Pixel-level annotation would significantly improve boundary accuracy and crack IoU.

**Crack IoU** — thin structures (1-3px wide) are inherently hard to score with IoU. A 1px spatial offset yields near-zero overlap. This is a fundamental metric limitation, not a model failure.

**Single-image inference only** — the API processes one image per request synchronously. Batch processing and async job queues are not implemented.

**Fixed 512x512 input** — very high-resolution images (4000x3000+) lose fine detail after downscaling. Sub-millimetre hairline cracks in such images may not be detected.

**No version management** — model checkpoints are overwritten in-place. Multiple experiment runs should use different `--out_dir` paths.

---

## 9. References

- SegFormer: Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021. https://arxiv.org/abs/2105.15203
- MiT-B2 pretrained weights: https://huggingface.co/nvidia/mit-b2
- Dice Loss: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 2016
- skmultilearn: http://scikit.ml/