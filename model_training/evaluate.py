"""
evaluate.py — Standalone evaluation on any split with qualitative outputs

Usage
    python evaluate.py --checkpoint ./checkpoints/best_model.pth
    python evaluate.py --checkpoint ./checkpoints/best_model.pth --save_visuals

Key functions in evaluate.py:
  - Console: per-class IoU, F1, mIoU, mF1 results table
  - eval_results.json
  - (optional) ./visuals/ — overlay images for qualitative analysis
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_dataloaders, CLASS_NAMES, CodebrimDataset
from model   import build_model, MultiLabelMetrics


CLASS_COLORS = {
    0: (255, 255, 100),   # efflorescence → yellow
    1: (255, 165,   0),   # corrosion     → orange
    2: (255,  70,  70),   # crack         → red
    3: ( 80, 130, 255),   # spalling      → blue
    4: ( 50, 200,  50),   # exposed bars  → green
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SegFormer-B2 on CODEBRIM")
    p.add_argument("--data_dir",    type=str, default="../roboflow_data",
                   help="Directory with images and XML annotations")
    p.add_argument("--checkpoint",  type=str, default="../checkpoints/best_model.pth",
                   help="Path to model checkpoint")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--threshold",   type=float, default=0.45)
    p.add_argument("--save_visuals",action="store_true",
                   help="Save overlay images for qualitative analysis")
    p.add_argument("--n_visuals",   type=int, default=20,
                   help="Number of images to save if --save_visuals")
    p.add_argument("--out_dir",     type=str, default="./eval_output")
    return p.parse_args()


def make_overlay(image_np: np.ndarray, pred_masks: np.ndarray,
                 gt_masks: np.ndarray, threshold: float) -> np.ndarray:
    """
    Build a side-by-side comparison: original | GT overlay | pred overlay.
    image_np : [H, W, 3] uint8 RGB
    pred_masks: [4, H, W] float32 probabilities
    gt_masks  : [4, H, W] float32 binary
    """
    h, w = image_np.shape[:2]
    orig  = image_np.copy()
    gt_v  = image_np.copy()
    pred_v= image_np.copy()

    for c in range(4):
        color = np.array(CLASS_COLORS[c], dtype=np.uint8)

        # GT overlay
        gt_bin = gt_masks[c] > 0.5
        gt_v[gt_bin] = (gt_v[gt_bin] * 0.4 + color * 0.6).astype(np.uint8)

        # Pred overlay
        pr_bin = pred_masks[c] >= threshold
        pred_v[pr_bin] = (pred_v[pr_bin] * 0.4 + color * 0.6).astype(np.uint8)

    # Add text labels
    for img, label in [(gt_v, "GT"), (pred_v, "PRED")]:
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend strip
    legend_h = 28
    legend   = np.zeros((legend_h, w * 3, 3), dtype=np.uint8)
    for c, name in enumerate(CLASS_NAMES):
        col = np.array(CLASS_COLORS[c], dtype=np.uint8)
        x0  = c * (w * 3 // 4)
        cv2.rectangle(legend, (x0, 4), (x0 + 16, 20), col.tolist(), -1)
        cv2.putText(legend, name, (x0 + 20, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    row     = np.concatenate([orig, gt_v, pred_v], axis=1)
    canvas  = np.concatenate([row, legend], axis=0)
    return canvas


def evaluate(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visuals"
    if args.save_visuals:
        vis_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Checkpoint: {args.checkpoint}")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = build_model(num_classes=5).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    print("[Eval] Model loaded.")

    # ── Data (use test split) ──────────────────────────────────────────────────
    _, _, test_loader, _ = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    metrics = MultiLabelMetrics(threshold=args.threshold)
    total_loss = 0.0
    n_saved    = 0

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with torch.amp.autocast(amp_device):
                outputs = model(pixel_values=images)

            logits_up = F.interpolate(
                outputs.logits, size=masks.shape[-2:],
                mode="bilinear", align_corners=False
            )
            probs = torch.sigmoid(logits_up)
            metrics.update(probs.cpu(), masks.cpu())

            # ── Save qualitative visuals ───────────────────────────────────────
            if args.save_visuals and n_saved < args.n_visuals:
                # Denormalise images for display
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                imgs_dn = (images.cpu() * std + mean).clamp(0, 1)

                for i in range(images.shape[0]):
                    if n_saved >= args.n_visuals:
                        break
                    img_np  = (imgs_dn[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pred_np = probs[i].cpu().numpy()    # [4, H, W]
                    gt_np   = masks[i].cpu().numpy()    # [4, H, W]

                    canvas  = make_overlay(img_np, pred_np, gt_np, args.threshold)
                    fname   = vis_dir / f"eval_{batch_idx:04d}_{i:02d}.jpg"
                    cv2.imwrite(str(fname), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                    n_saved += 1

    # ── Results table ──────────────────────────────────────────────────────────
    m = metrics.compute()

    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS  (threshold={args.threshold})")
    print(f"{'='*55}")
    print(f"  {'Class':<14} {'IoU':>10} {'F1':>10}")
    print(f"  {'-'*34}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<14} {m['iou'][i]:>10.4f} {m['f1'][i]:>10.4f}")
    print(f"  {'-'*34}")
    print(f"  {'MEAN':<14} {m['miou']:>10.4f} {m['mf1']:>10.4f}")
    print(f"{'='*55}")

    if args.save_visuals:
        print(f"\n  Visuals saved to: {vis_dir}  ({n_saved} images)")

    # Save JSON
    results = {
        "checkpoint": args.checkpoint,
        "threshold":  args.threshold,
        **m,
        "per_class": {
            name: {"iou": m["iou"][i], "f1": m["f1"][i]}
            for i, name in enumerate(CLASS_NAMES)
        }
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    evaluate(parse_args())