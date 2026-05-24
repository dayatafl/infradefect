"""
app.py — Infrastructure Defect Segmentation API
================================================
This is the main application file that ties together:
  - The trained CODEBRIM SegFormer-B2 model (from train.py)
  - FastAPI routes for upload → inference → result serving
  - Synthetic demo fallback when no checkpoint is present

The trained checkpoint (best_model.pth from train.py) should be placed at:
    ./checkpoints/best_model.pth

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Run in Docker:
    docker compose up
"""

import os
import uuid
import time
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()   # load .env from project root if present

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import SegformerForSemanticSegmentation

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("app")

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent   # project root (one level up from backend/)
UPLOAD_DIR     = Path(os.getenv("UPLOAD_DIR",     str(BASE_DIR / "uploads")))
RESULT_DIR     = Path(os.getenv("RESULT_DIR",     str(BASE_DIR / "results")))
CHECKPOINT     = Path(os.getenv("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints" / "best_model.pth")))
INPUT_SIZE     = int(os.getenv("INPUT_SIZE",     "512"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))
NUM_CLASSES    = int(os.getenv("NUM_CLASSES",    "5"))

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Class schema (matches dataset.py CLASS_NAMES)
CLASS_NAMES = ["efflorescence", "corrosion", "crack", "spalling", "exposed_bars"]
CLASS_COLORS = [
    (255, 255, 100),   # efflorescence → yellow
    (255, 165,   0),   # corrosion     → orange
    (255,  70,  70),   # crack         → red
    ( 80, 130, 255),   # spalling      → blue
    ( 50, 200,  50),   # exposed bars  → green
]
CLASS_HEX = ["#ffff64", "#ffa500", "#ff4646", "#5082ff", "#32c832"]

# Severity thresholds (% of total image pixels)
SEVERITY = [
    (0.0,  "none"),
    (0.5,  "low"),
    (2.0,  "medium"),
    (8.0,  "high"),
    (float("inf"), "critical"),
]


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ClassStat(BaseModel):
    pixel_count:  int
    percentage:   float
    color_hex:    str
    severity:     str

class PredictionResponse(BaseModel):
    job_id:        str
    overlay_url:   str
    mask_url:      str
    overlay_b64:   str          # base64 PNG — lets frontend render without CORS
    mask_b64:      str          # base64 PNG of raw class-colour mask
    class_stats:   dict         # {class_name: ClassStat}
    inference_ms:  float
    engine_mode:   str          # "model" | "demo"
    image_size:    list         # [width, height]
    defects_found: list         # names of classes with severity != "none"

class HealthResponse(BaseModel):
    status:       str
    engine_mode:  str
    model_name:   str
    device:       str
    checkpoint:   Optional[str]


# ── Image normalisation helpers ────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image) -> torch.Tensor:
    """PIL RGB → normalised float tensor [1, 3, 512, 512]."""
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W]
    return tensor


def build_class_stats(prob_masks: np.ndarray, h: int, w: int) -> dict:
    """
    prob_masks : [4, H, W] float32 in [0,1]
    Returns dict of {class_name: ClassStat fields as dict}
    """
    total = h * w
    stats = {}
    for i, name in enumerate(CLASS_NAMES):
        binary      = (prob_masks[i] >= CONF_THRESHOLD).astype(np.uint8)
        # Morphological cleanup: remove blobs < 50px²
        n_lbl, lbl, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean = np.zeros_like(binary)
        for j in range(1, n_lbl):
            if st[j, cv2.CC_STAT_AREA] >= 50:
                clean[lbl == j] = 1

        px_count = int(clean.sum())
        pct      = round(px_count / total * 100, 3)

        severity = "none"
        for threshold, label in SEVERITY:
            if pct <= threshold:
                severity = label
                break

        stats[name] = {
            "pixel_count": px_count,
            "percentage":  pct,
            "color_hex":   CLASS_HEX[i],
            "severity":    severity,
        }
    return stats


def build_overlay(original: np.ndarray, prob_masks: np.ndarray) -> np.ndarray:
    """
    Blend colour-coded defect masks over the original image.
    original   : [H, W, 3] uint8 RGB
    prob_masks : [4, H, W] float32
    Returns    : [H, W, 3] uint8 RGB overlay
    """
    overlay = original.copy().astype(np.float32)
    for i, color in enumerate(CLASS_COLORS):
        binary = (prob_masks[i] >= CONF_THRESHOLD).astype(np.uint8)
        # Morphological cleanup
        n_lbl, lbl, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean = np.zeros_like(binary)
        for j in range(1, n_lbl):
            if st[j, cv2.CC_STAT_AREA] >= 50:
                clean[lbl == j] = 1
        if clean.sum() == 0:
            continue
        color_arr = np.array(color, dtype=np.float32)
        mask_3ch  = clean[:, :, np.newaxis].astype(np.float32)
        overlay   = overlay * (1 - 0.5 * mask_3ch) + color_arr * 0.5 * mask_3ch

    return np.clip(overlay, 0, 255).astype(np.uint8)


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Inference engines ──────────────────────────────────────────────────────────

class ModelEngine:
    """Real SegFormer-B2 inference using the trained CODEBRIM checkpoint."""
    mode       = "model"
    model_name = "SegFormer-B2 (CODEBRIM fine-tuned)"

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading checkpoint on {self.device}: {checkpoint_path}")

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # Support both raw state_dict and full training checkpoint
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        logger.info("ModelEngine ready.")

    def predict(self, image: Image.Image) -> dict:
        orig_w, orig_h = image.size
        tensor = preprocess(image).to(self.device)

        with torch.no_grad():
            outputs   = self.model(pixel_values=tensor)
            logits_up = F.interpolate(
                outputs.logits,
                size=(orig_h, orig_w),
                mode="bilinear", align_corners=False
            )
            probs = torch.sigmoid(logits_up).squeeze(0).cpu().numpy()  # [4, H, W]

        orig_np   = np.array(image.convert("RGB"))
        overlay_np = build_overlay(orig_np, probs)
        stats      = build_class_stats(probs, orig_h, orig_w)

        return {
            "overlay": Image.fromarray(overlay_np),
            "mask":    _probs_to_color_mask(probs, orig_h, orig_w),
            "stats":   stats,
            "size":    (orig_w, orig_h),
        }



# ── Procedural defect drawing (demo mode only) ─────────────────────────────────

def _draw_crack_probs(canvas, rng, w, h, thickness=2.5):
    x  = float(rng.integers(w // 4, 3 * w // 4))
    y  = float(rng.integers(h // 4, 3 * h // 4))
    angle  = rng.uniform(0, 2 * np.pi)
    length = rng.integers(80, min(w, h) // 2)
    t = thickness
    for _ in range(length):
        angle += rng.uniform(-0.25, 0.25)
        x += np.cos(angle) * 2
        y += np.sin(angle) * 2
        xi, yi = int(x), int(y)
        r = max(1, int(t))
        ys_g, xs_g = np.ogrid[-r:r+1, -r:r+1]
        circle = xs_g**2 + ys_g**2 <= r**2
        y0, y1 = max(0, yi-r), min(h, yi+r+1)
        x0, x1 = max(0, xi-r), min(w, xi+r+1)
        if y1 > y0 and x1 > x0:
            cy0, cy1 = y0-(yi-r), y1-(yi-r)
            cx0, cx1 = x0-(xi-r), x1-(xi-r)
            canvas[y0:y1, x0:x1][circle[cy0:cy1, cx0:cx1]] = 0.9
        t = max(1.0, t - 0.015)


def _draw_blob_probs(canvas, rng, w, h, max_r=60):
    cx   = int(rng.integers(max_r, w - max_r))
    cy   = int(rng.integers(max_r, h - max_r))
    n    = 10
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii  = rng.uniform(max_r * 0.4, max_r, size=n)
    ys, xs = np.mgrid[0:h, 0:w]
    amap   = np.arctan2(ys - cy, xs - cx)
    bins   = ((amap + np.pi) / (2*np.pi / n)).astype(int) % n
    rdist  = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    canvas[rdist < radii[bins]] = 0.85


def _draw_deformation_probs(canvas, rng, w, h):
    thick = rng.integers(18, 38)
    axis  = rng.choice(["h", "v"])
    if axis == "h":
        y0 = rng.integers(h//3, 2*h//3)
        canvas[y0:y0+thick, w//5:4*w//5] = 0.8
    else:
        x0 = rng.integers(w//3, 2*w//3)
        canvas[h//5:4*h//5, x0:x0+thick] = 0.8


def _probs_to_color_mask(probs: np.ndarray, h: int, w: int) -> Image.Image:
    """Convert [4, H, W] probabilities to a coloured mask PIL image."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)  # dark background
    for i, color in enumerate(CLASS_COLORS):
        canvas[probs[i] >= CONF_THRESHOLD] = color
    return Image.fromarray(canvas)


# ── Engine factory ─────────────────────────────────────────────────────────────

def get_engine():
    if CHECKPOINT.exists():
        try:
            return ModelEngine(str(CHECKPOINT))
        except Exception as e:
            logger.warning(f"Failed to load checkpoint ({e}); using demo mode.")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="InfraDefect Segmentation API",
    description=(
        "Pixel-level segmentation of infrastructure defects: "
        "cracks, spalling, corrosion, deformation. "
        "Built on SegFormer-B2 fine-tuned on CODEBRIM."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve result images as static files (used by overlay_url / mask_url)
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")

# ── Global engine (loaded once at startup) ────────────────────────────────────
_engine = None


@app.on_event("startup")
async def startup():
    global _engine
    logger.info("Starting up — loading inference engine…")
    _engine = get_engine()
    logger.info(f"Engine ready: mode={_engine.mode}, model={_engine.model_name}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Liveness + readiness probe."""
    return HealthResponse(
        status      = "ok",
        engine_mode = _engine.mode if _engine else "not_loaded",
        model_name  = _engine.model_name if _engine else "N/A",
        device      = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available()
                      else "cpu",
        checkpoint  = str(CHECKPOINT) if CHECKPOINT.exists() else None,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accept a JPEG or PNG upload and return segmentation results.

    Response includes:
    - overlay_url   : path to colour-coded overlay PNG (served as static file)
    - mask_url      : path to raw class-colour mask PNG
    - overlay_b64   : base64-encoded overlay PNG (avoids CORS issues in browser)
    - class_stats   : per-class pixel count, % coverage, severity level
    - defects_found : list of class names with at least low severity
    - inference_ms  : wall-clock inference time
    - engine_mode   : 'model' (trained weights) or 'demo' (synthetic)
    """
    if _engine is None:
        raise HTTPException(503, "Inference engine not loaded yet.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            415,
            f"Unsupported file type '{file.content_type}'. Upload JPEG or PNG."
        )

    # ── Load image ─────────────────────────────────────────────────────────────
    try:
        raw   = await file.read()
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image. Ensure the file is a valid JPEG or PNG.")

    # ── Run inference ──────────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"[{job_id}] Received {file.filename} ({image.size[0]}×{image.size[1]})")

    t0 = time.perf_counter()
    try:
        result = _engine.predict(image)
    except Exception as exc:
        logger.exception(f"[{job_id}] Inference error: {exc}")
        raise HTTPException(500, f"Inference failed: {str(exc)}")
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Save output images ─────────────────────────────────────────────────────
    overlay_name = f"{job_id}_overlay.png"
    mask_name    = f"{job_id}_mask.png"
    result["overlay"].save(str(RESULT_DIR / overlay_name))
    result["mask"].save(str(RESULT_DIR / mask_name))

    # ── Build response ─────────────────────────────────────────────────────────
    defects_found = [
        name for name, stat in result["stats"].items()
        if stat["severity"] not in ("none",)
    ]

    logger.info(
        f"[{job_id}] Done {inference_ms}ms | "
        f"defects={defects_found} | mode={_engine.mode}"
    )

    return PredictionResponse(
        job_id        = job_id,
        overlay_url   = f"/results/{overlay_name}",
        mask_url      = f"/results/{mask_name}",
        overlay_b64   = image_to_b64(result["overlay"]),
        mask_b64      = image_to_b64(result["mask"]),
        class_stats   = result["stats"],
        inference_ms  = inference_ms,
        engine_mode   = _engine.mode,
        image_size    = list(result["size"]),
        defects_found = defects_found,
    )


@app.get("/api/model-info")
async def model_info():
    """Metadata about the currently loaded model."""
    if _engine is None:
        raise HTTPException(503, "Engine not loaded.")
    return {
        "model_name":  _engine.model_name,
        "mode":        _engine.mode,
        "num_classes": NUM_CLASSES,
        "input_size":  INPUT_SIZE,
        "threshold":   CONF_THRESHOLD,
        "classes":     CLASS_NAMES,
        "colors":      CLASS_HEX,
    }


@app.get("/api/classes")
async def classes():
    """Return class schema for the frontend legend."""
    return [
        {"index": i, "name": CLASS_NAMES[i], "color": CLASS_HEX[i]}
        for i in range(NUM_CLASSES)
    ]


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)