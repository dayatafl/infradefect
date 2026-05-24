"""
train.py — Full training pipeline for CODEBRIM defect segmentation

Usage
-----
    python train.py --data_dir ./data/codebrim --batch_size 4
    python train.py --data_dir ./data/codebrim --resume ./checkpoints/last.pth

What this script does vs gem/train.py
--------------------------------------
  gem/train.py                          This script
  ─────────────────────────────         ──────────────────────────────────────
  val loss only for checkpointing  →    val mIoU (task metric) for checkpoint
  no per-class metrics             →    per-class IoU + F1 every epoch
  no resume support                →    full resume from last.pth
  hardcoded pos_weight [3,5,5,6]   →    data-driven pos_weight from frequencies
  no gradient clipping             →    gradient clipping (max_norm=1.0)
  no warmup LR schedule            →    linear warmup + cosine decay (S2)
  saves state_dict only            →    saves full training state
  prints loss only                 →    rich epoch summary table
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim

from dataset import get_dataloaders, CLASS_NAMES
from model   import build_model, CodebrimLoss, MultiLabelMetrics, EarlyStopping


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train SegFormer-B2 on CODEBRIM")
    p.add_argument("--data_dir",     type=str,   default="../roboflow_data",
                   help="Directory with images and XML annotations")
    p.add_argument("--out_dir",      type=str,   default="../checkpoints",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--warmup_epochs",type=int,   default=5,
                   help="Stage 1: decoder-only warmup epochs")
    p.add_argument("--main_epochs",  type=int,   default=45,
                   help="Stage 2: full fine-tuning epochs")
    p.add_argument("--num_workers",  type=int,   default=None,
                   help="DataLoader workers (default: 4 on Linux, 2 on Windows)")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--threshold",    type=float, default=0.45,
                   help="Sigmoid threshold for binary prediction")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int,
                    best_miou: float, metrics: dict, config: dict):
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "scheduler_state":scheduler.state_dict() if scheduler else None,
        "best_miou":      best_miou,
        "metrics":        metrics,
        "config":         config,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


# ── Linear warmup LR scheduler ────────────────────────────────────────────────

class LinearWarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linearly ramp LR from 0 → base_lr over `warmup_steps` steps,
    then apply cosine annealing over the remaining steps.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = (step + 1) / max(self.warmup_steps, 1)
        else:
            import math
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            factor   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# ── Pretty epoch summary ───────────────────────────────────────────────────────

def print_epoch_summary(stage: str, epoch: int, total_epochs: int,
                        train_loss: float, val_loss: float,
                        metrics: dict, elapsed: float, is_best: bool):
    tag = "★ BEST" if is_best else ""
    print(f"\n{'─'*70}")
    print(f"  {stage} | Epoch {epoch:>3}/{total_epochs}  ({elapsed:.0f}s)  {tag}")
    print(f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}")
    print(f"  mIoU: {metrics['miou']:.4f}   mF1: {metrics['mf1']:.4f}")
    print(f"  {'Class':<14} {'IoU':>8} {'F1':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<14} {metrics['iou'][i]:>7.4f} {metrics['f1'][i]:>7.4f}")
    print(f"{'─'*70}")


# ── Training loop ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device,
              metrics_tracker: MultiLabelMetrics, is_train: bool):
    """One full pass over the data. Returns average loss."""
    model.train() if is_train else model.eval()
    total_loss = 0.0
    metrics_tracker.reset()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    with ctx:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with torch.amp.autocast(amp_device):
                outputs = model(pixel_values=images)
                loss    = criterion(outputs.logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # Gradient clipping: prevents exploding gradients in fine-tuning
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

            # Accumulate metrics (CPU to avoid GPU memory pressure)
            with torch.no_grad():
                logits_up = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )
                probs = torch.sigmoid(logits_up).cpu()
                metrics_tracker.update(probs, masks.cpu())

    return total_loss / len(loader)


# ── Main training function ─────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )

    # ── Model & loss ───────────────────────────────────────────────────────────
    model     = build_model(num_classes=5).to(device)
    criterion = CodebrimLoss(class_weights=class_weights).to(device)
    scaler    = torch.amp.GradScaler(device.type)
    metrics_train = MultiLabelMetrics(threshold=args.threshold)
    metrics_val   = MultiLabelMetrics(threshold=args.threshold)
    early_stop    = EarlyStopping(patience=10, mode="max")  # watch mIoU

    config = vars(args)
    best_miou   = 0.0
    start_epoch = 0
    log_entries = []

    # ── Resume support ─────────────────────────────────────────────────────────
    skip_stage1 = False
    if args.resume and Path(args.resume).exists():
        print(f"[Train] Resuming from {args.resume}")
        start_epoch, best_miou = load_checkpoint(
            args.resume, model, device=device
        )
        print(f"[Train] Resumed at epoch {start_epoch}, best mIoU {best_miou:.4f}")
        # If resuming from Stage 2, skip Stage 1
        if start_epoch > args.warmup_epochs:
            skip_stage1 = True
            print(f"[Train] Skipping Stage 1 (already completed)")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Decoder warmup (encoder frozen)
    # ══════════════════════════════════════════════════════════════════════════
    if not skip_stage1:
        print(f"\n{'='*70}")
        print(f"  STAGE 1 — Decoder warmup ({args.warmup_epochs} epochs, encoder frozen)")
        print(f"{'='*70}")

        # Freeze everything except the decode_head
        for name, param in model.named_parameters():
            param.requires_grad = "decode_head" in name

        trainable_s1 = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable params: {sum(p.numel() for p in trainable_s1):,} "
              f"(decode_head only)")

        optimizer_s1 = optim.AdamW(trainable_s1, lr=6e-4, weight_decay=0.01)
        scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_s1, T_max=args.warmup_epochs, eta_min=1e-5
        )

        for epoch in range(1, args.warmup_epochs + 1):
            t0 = time.time()
            train_loss = run_epoch(
                model, train_loader, criterion, optimizer_s1, scaler,
                device, metrics_train, is_train=True
            )
            val_loss = run_epoch(
                model, val_loader, criterion, optimizer_s1, scaler,
                device, metrics_val, is_train=False
            )
            scheduler_s1.step()

            m = metrics_val.compute()
            elapsed = time.time() - t0
            print(f"  S1 Epoch {epoch:>2}/{args.warmup_epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"mIoU={m['miou']:.4f}  ({elapsed:.0f}s)")

        # Save stage-1 checkpoint
        save_checkpoint(
            str(out_dir / "stage1_final.pth"),
            model, optimizer_s1, scheduler_s1,
            epoch=args.warmup_epochs, best_miou=best_miou,
            metrics=metrics_val.compute(), config=config
        )
        print("  [Stage 1 complete] checkpoint saved → stage1_final.pth")
    else:
        print(f"\n[Train] Skipping Stage 1 (resuming from epoch {start_epoch})")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Full fine-tuning with layer-wise LR
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STAGE 2 — Full fine-tuning ({args.main_epochs} epochs)")
    print(f"{'='*70}")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {total_params:,} (all layers)")

    # Layer-wise learning rates:
    #   encoder (MiT-B2 backbone)  → 6e-5  (small: don't destroy ImageNet features)
    #   decoder (decode_head)       → 6e-4  (10× larger: fast adaptation)
    encoder_params = [p for n, p in model.named_parameters() if "decode_head" not in n]
    decoder_params = [p for n, p in model.named_parameters() if "decode_head" in n]

    optimizer_s2 = optim.AdamW([
        {"params": encoder_params, "lr": 6e-5},
        {"params": decoder_params, "lr": 6e-4},
    ], weight_decay=0.01)

    # Linear warmup (first 10% of steps) + cosine decay
    total_steps   = args.main_epochs * len(train_loader)
    warmup_steps  = int(0.10 * total_steps)
    scheduler_s2  = LinearWarmupCosineScheduler(
        optimizer_s2, warmup_steps=warmup_steps, total_steps=total_steps
    )
    # Step scheduler per batch (not per epoch) for smooth warmup
    scheduler_step_per_batch = True

    print(f"  Scheduler: {warmup_steps} warmup steps → cosine over {total_steps} steps")

    # If resuming into Stage 2, reload optimizer and scheduler state
    if args.resume and skip_stage1 and Path(args.resume).exists():
        print(f"[Train] Restoring optimizer state from checkpoint...")
        load_checkpoint(
            args.resume, model, optimizer_s2, scheduler_s2, device=device
        )

    # ── Stage 2 training loop ──────────────────────────────────────────────────
    stage2_start_epoch = start_epoch + 1 if skip_stage1 else 1
    
    for epoch in range(stage2_start_epoch, args.main_epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        metrics_train.reset()
        amp_device = "cuda" if device.type == "cuda" else "cpu"

        for step, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with torch.amp.autocast(amp_device):
                outputs = model(pixel_values=images)
                loss    = criterion(outputs.logits, masks)

            optimizer_s2.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_s2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer_s2)
            scaler.update()

            if scheduler_step_per_batch:
                scheduler_s2.step()

            total_train_loss += loss.item()

            # Update train metrics
            with torch.no_grad():
                logits_up = torch.nn.functional.interpolate(
                    outputs.logits, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )
                metrics_train.update(torch.sigmoid(logits_up).cpu(), masks.cpu())

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        metrics_val.reset()

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)
                with torch.amp.autocast(amp_device):
                    outputs  = model(pixel_values=images)
                    val_loss = criterion(outputs.logits, masks)
                total_val_loss += val_loss.item()
                logits_up = torch.nn.functional.interpolate(
                    outputs.logits, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )
                metrics_val.update(torch.sigmoid(logits_up).cpu(), masks.cpu())

        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics  = metrics_val.compute()
        elapsed      = time.time() - t0

        # ── Checkpoint: save best by mIoU ──────────────────────────────────────
        is_best = val_metrics["miou"] > best_miou
        if is_best:
            best_miou = val_metrics["miou"]
            save_checkpoint(
                str(out_dir / "best_model.pth"),
                model, optimizer_s2, scheduler_s2, epoch,
                best_miou, val_metrics, config
            )

        # Always save last checkpoint for resume
        save_checkpoint(
            str(out_dir / "last.pth"),
            model, optimizer_s2, scheduler_s2, epoch,
            best_miou, val_metrics, config
        )

        # ── Logging ────────────────────────────────────────────────────────────
        print_epoch_summary(
            "Stage 2", epoch, args.main_epochs,
            avg_train_loss, avg_val_loss,
            val_metrics, elapsed, is_best
        )

        log_entry = {
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_loss":   round(avg_val_loss,   5),
            "miou":       round(val_metrics["miou"], 5),
            "mf1":        round(val_metrics["mf1"],  5),
            "iou":        [round(v, 5) for v in val_metrics["iou"]],
            "f1":         [round(v, 5) for v in val_metrics["f1"]],
            "is_best":    is_best,
        }
        log_entries.append(log_entry)

        # Flush log to disk each epoch (safe for long runs / crashes)
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(log_entries, f, indent=2)

        # ── Early stopping check ───────────────────────────────────────────────
        if early_stop.step(val_metrics["miou"]):
            print(f"\n[EarlyStopping] No mIoU improvement for {early_stop.patience} epochs. "
                  f"Stopping at epoch {epoch}.")
            break

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION on held-out test set
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  FINAL EVALUATION — held-out test set")
    print(f"{'='*70}")

    # Load best checkpoint
    best_ckpt = str(out_dir / "best_model.pth")
    if Path(best_ckpt).exists():
        model_eval = build_model(num_classes=5).to(device)
        load_checkpoint(best_ckpt, model_eval, device=device)
        model_eval.eval()
    else:
        model_eval = model
        model_eval.eval()

    test_metrics = MultiLabelMetrics(threshold=args.threshold)
    test_loss_total = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            with torch.amp.autocast(device.type):
                outputs   = model_eval(pixel_values=images)
                test_loss = criterion(outputs.logits, masks)
            test_loss_total += test_loss.item()
            logits_up = torch.nn.functional.interpolate(
                outputs.logits, size=masks.shape[-2:],
                mode="bilinear", align_corners=False
            )
            test_metrics.update(torch.sigmoid(logits_up).cpu(), masks.cpu())

    tm = test_metrics.compute()
    avg_test_loss = test_loss_total / len(test_loader)

    print(f"\n  Test loss : {avg_test_loss:.4f}")
    print(f"  Test mIoU : {tm['miou']:.4f}")
    print(f"  Test mF1  : {tm['mf1']:.4f}")
    print(f"\n  {'Class':<14} {'IoU':>8} {'F1':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<14} {tm['iou'][i]:>7.4f} {tm['f1'][i]:>7.4f}")

    # Save test results
    with open(out_dir / "test_results.json", "w") as f:
        json.dump({"test_loss": avg_test_loss, **tm}, f, indent=2)

    print(f"\n[Done] Best val mIoU: {best_miou:.4f}")
    print(f"[Done] Checkpoints saved to: {out_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)