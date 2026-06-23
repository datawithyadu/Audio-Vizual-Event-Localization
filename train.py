"""
Training script for the AVE att_Net model.

Usage:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
import utils
from dataset import AVEDataset, AVEDatasetH5
from models import AVEModel
import evaluate as ev

_H5_AVAILABLE = (
    os.path.exists(config.AUDIO_H5_FILE) and
    os.path.exists(config.VISUAL_H5_FILE)
)


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# One training epoch
# ──────────────────────────────────────────────

def train_epoch(
    model: AVEModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    modality_dropout_prob: float = config.MODALITY_DROPOUT_PROB,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        audio  = batch["audio"].to(device)    # (B, 10, 128)
        video  = batch["video"].to(device)    # (B, 10, 49, 512)
        labels = batch["labels"].to(device)   # (B, 10)

        optimizer.zero_grad()
        logits = model(audio, video, modality_dropout_prob=modality_dropout_prob)
        # logits: (B, T, 29)  →  flatten to (B*T, 29) for cross-entropy
        loss = criterion(logits.reshape(-1, config.NUM_CLASSES), labels.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)   # (B, T)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds_cat  = torch.cat([p.reshape(-1) for p in all_preds])
    labels_cat = torch.cat([l.reshape(-1) for l in all_labels])

    return {
        "loss"    : total_loss / len(loader),
        "accuracy": ev.per_second_accuracy(preds_cat, labels_cat),
        "recall"  : ev.per_second_recall(preds_cat, labels_cat),
    }


# ──────────────────────────────────────────────
# Validation / test epoch (no gradient)
# ──────────────────────────────────────────────

def eval_epoch(
    model: AVEModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            audio  = batch["audio"].to(device)
            video  = batch["video"].to(device)
            labels = batch["labels"].to(device)

            logits = model(audio, video, modality_dropout_prob=0.0)
            loss   = criterion(logits.reshape(-1, config.NUM_CLASSES), labels.reshape(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds_cat  = torch.cat([p.reshape(-1) for p in all_preds])
    labels_cat = torch.cat([l.reshape(-1) for l in all_labels])

    return {
        "loss"    : total_loss / len(loader),
        "accuracy": ev.per_second_accuracy(preds_cat, labels_cat),
        "recall"  : ev.per_second_recall(preds_cat, labels_cat),
    }


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    patience: int = config.EARLY_STOPPING_PATIENCE,
) -> AVEModel:
    device = get_device()
    print(f"Device : {device}")

    utils.ensure_dirs()

    # ── datasets ──────────────────────────────
    if _H5_AVAILABLE:
        train_set = AVEDatasetH5("train")
        val_set   = AVEDatasetH5("val")
    else:
        train_set = AVEDataset(split="train", use_preextracted=True)
        val_set   = AVEDataset(split="val",   use_preextracted=True)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    print(f"Train  : {len(train_set)} clips  ({len(train_loader)} batches)")
    print(f"Val    : {len(val_set)} clips")

    # ── class weights (Guide §5 — class imbalance) ────────
    train_samples = utils.load_split_file(config.TRAIN_SET_FILE)
    class_weights = utils.compute_class_weights(train_samples).to(device)

    # ── model ─────────────────────────────────
    model = AVEModel().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    no_improve    = 0
    best_ckpt     = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"train loss {train_metrics['loss']:.4f}  acc {train_metrics['accuracy']:.3f}  "
            f"rec {train_metrics['recall']:.3f} | "
            f"val loss {val_metrics['loss']:.4f}  acc {val_metrics['accuracy']:.3f}  "
            f"rec {val_metrics['recall']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve    = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest model saved to: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, weights_only=True, map_location=device))
    return model


if __name__ == "__main__":
    trained_model = train()
