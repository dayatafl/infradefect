"""
app.py — Infrastructure Defect Segmentation API
Adapted for HuggingFace Spaces (CPU inference)

Changes from local version:
- Checkpoint downloaded from HuggingFace Hub at startup (not from local disk)
- CPU-only inference (no CUDA checks that would fail on HF Spaces free tier)
- CORS configured for Vercel frontend domain
- Startup downloads checkpoint if not cached
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
load_dotenv()

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import SegformerForSemanticSegmentation

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("app")

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "uploads"
RESULT_DIR  = BASE_DIR / "results"

# Checkpoint: downloaded from HuggingFace Hub at startup
# Set HF_MODEL_REPO in Space secrets to your model repo, e.g. "yourusername/infradefect-model"
HF_MODEL_REPO   = os.getenv("HF_MODEL_REPO", "")
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pth"

INPUT_SIZE     = int(os.getenv("INPUT_SIZE",     "512"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))
NUM_CLASSES    = int(os.getenv("NUM_CLASSES",    "5"))

# Frontend origin — set in HF Space secrets
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
(BASE_DIR / "checkpoints").mkdir(exist_ok=True)

# Class schema
CLASS_NAMES  = ["efflorescence", "corrosion", "crack", "spalling", "exposed_bars"]
CLASS_COLORS = [
    (255, 255, 100),
    (255, 165,   0),
    (255,  70,  70),
    ( 80, 130, 255),
    ( 50, 200,  50),
]
CLASS_HEX = ["#ffff64", "#ffa500", "#ff4646", "#5082ff", "#32c832"]

SEVERITY = [
    (0.0,          "none"),
    (0.5,          "low"),
    (2.0,          "medium"),
    (8.0,          "high"),
    (float("inf"), "critical"),
]

# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ClassStat(BaseModel):
    pixel_count: int
    percentage:  float
    color_hex:   str
    severity:    str

class PredictionResponse(BaseModel):
    job_id:        str
    overlay_url:   str
    mask_url:      str
    overlay_b64:   str
    class_stats:   dict
    inference_ms:  float
    engine_mode:   str
    image_size:    list
    defects_found: list

class HealthResponse(BaseModel):
    status:      str
    engine_mode: str
    model_name:  str
    device:      str
    checkpoint:  Optional[str]

# ── Image helpers ──────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def build_class_stats(prob_masks: np.ndarray, h: int, w: int) -> dict:
    total = h * w
    stats = {}
    for i, name in enumerate(CLASS_NAMES):
        binary = (prob_masks[i] >= CONF_THRESHOLD).astype(np.uint8)
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
    overlay = original.copy().astype(np.float32)
    for i, color in enumerate(CLASS_COLORS):
        binary = (prob_masks[i] >= CONF_THRESHOLD).astype(np.uint8)
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


# ── Checkpoint download ────────────────────────────────────────────────────────

def download_checkpoint():
    """Download best_model.pth from HuggingFace Hub if not already cached."""
    if CHECKPOINT_PATH.exists():
        logger.info(f"Checkpoint already cached at {CHECKPOINT_PATH}")
        return True

    if not HF_MODEL_REPO:
        logger.error("HF_MODEL_REPO not set and no local checkpoint found.")
        return False

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"Downloading checkpoint from {HF_MODEL_REPO}...")
        local = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="best_model.pth",
            local_dir=str(BASE_DIR / "checkpoints"),
        )
        logger.info(f"Checkpoint downloaded to {local}")
        return True
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        return False


# ── Inference engine ───────────────────────────────────────────────────────────

class ModelEngine:
    mode       = "model"
    model_name = "SegFormer-B2 (CODEBRIM fine-tuned)"

    def __init__(self, checkpoint_path: str):
        # Always CPU on HuggingFace Spaces free tier
        self.device = torch.device("cpu")
        logger.info(f"Loading model on CPU from {checkpoint_path}")

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("ModelEngine ready (CPU).")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        orig_w, orig_h = image.size
        tensor = preprocess(image)

        outputs  = self.model(pixel_values=tensor)
        logits   = F.interpolate(
            outputs.logits,
            size=(orig_h, orig_w),
            mode="bilinear", align_corners=False
        )
        probs = torch.sigmoid(logits).squeeze(0).numpy()   # [5, H, W]

        orig_arr = np.array(image.convert("RGB"))
        stats    = build_class_stats(probs, orig_h, orig_w)
        overlay  = build_overlay(orig_arr, probs)

        # Build colour mask
        mask_arr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        mask_arr[:] = (20, 20, 20)
        for i, color in enumerate(CLASS_COLORS):
            mask_arr[probs[i] >= CONF_THRESHOLD] = color

        return {
            "overlay": Image.fromarray(overlay),
            "mask":    Image.fromarray(mask_arr),
            "stats":   stats,
            "size":    (orig_w, orig_h),
        }


# ── Engine factory ─────────────────────────────────────────────────────────────

def get_engine():
    ok = download_checkpoint()
    if not ok or not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            "No checkpoint available. Set HF_MODEL_REPO in Space secrets "
            "or upload best_model.pth to the Space."
        )
    return ModelEngine(str(CHECKPOINT_PATH))


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="InfraDefect Segmentation API",
    description="Pixel-level segmentation of infrastructure defects using SegFormer-B2.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tightened once Vercel URL is known
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")

_engine = None


@app.on_event("startup")
async def startup():
    global _engine
    logger.info("Starting inference engine...")
    _engine = get_engine()
    logger.info(f"Engine ready: {_engine.model_name}")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status      = "ok",
        engine_mode = _engine.mode if _engine else "not_loaded",
        model_name  = _engine.model_name if _engine else "N/A",
        device      = "cpu",
        checkpoint  = str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _engine is None:
        raise HTTPException(503, "Inference engine not loaded.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(415, f"Unsupported file type. Upload JPEG or PNG.")

    try:
        raw   = await file.read()
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    job_id = str(uuid.uuid4())[:8]
    logger.info(f"[{job_id}] {file.filename} ({image.size[0]}×{image.size[1]})")

    t0 = time.perf_counter()
    try:
        result = _engine.predict(image)
    except Exception as exc:
        logger.exception(f"[{job_id}] Inference error: {exc}")
        raise HTTPException(500, f"Inference failed: {str(exc)}")
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    overlay_name = f"{job_id}_overlay.png"
    mask_name    = f"{job_id}_mask.png"
    result["overlay"].save(str(RESULT_DIR / overlay_name))
    result["mask"].save(str(RESULT_DIR / mask_name))

    defects_found = [
        name for name, stat in result["stats"].items()
        if stat["severity"] != "none"
    ]

    logger.info(f"[{job_id}] Done {inference_ms}ms | defects={defects_found}")

    return PredictionResponse(
        job_id        = job_id,
        overlay_url   = f"/results/{overlay_name}",
        mask_url      = f"/results/{mask_name}",
        overlay_b64   = image_to_b64(result["overlay"]),
        class_stats   = result["stats"],
        inference_ms  = inference_ms,
        engine_mode   = _engine.mode,
        image_size    = list(result["size"]),
        defects_found = defects_found,
    )


@app.get("/api/model-info")
async def model_info():
    if _engine is None:
        raise HTTPException(503, "Engine not loaded.")
    return {
        "model_name": _engine.model_name,
        "mode":       _engine.mode,
        "num_classes": NUM_CLASSES,
        "input_size":  INPUT_SIZE,
        "threshold":   CONF_THRESHOLD,
        "classes":     CLASS_NAMES,
        "colors":      CLASS_HEX,
    }


@app.get("/api/classes")
async def classes():
    return [
        {"index": i, "name": CLASS_NAMES[i], "color": CLASS_HEX[i]}
        for i in range(NUM_CLASSES)
    ]
