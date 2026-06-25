"""
Multi-seed comparison: VGG19 h5 features vs self-extracted R(2+1)D .pt features.
Runs 3 seeds × 2 feature sources = 6 training runs total.

Usage:
    python run_multiseed.py

Outputs (in project root):
    results_vgg19_seed{S}.txt          per-run detailed report
    results_r2plus1d_seed{S}.txt       per-run detailed report
    results_seed_comparison.md         summary table with mean ± std
    multiseed.log                      full epoch-level log
"""

import os, sys, time, logging, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, utils
from models import AVEModel
from dataset import AVEDataset, AVEDatasetH5
import evaluate as ev

# ── logging ──────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiseed.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [42, 123, 2024]
ROOT   = os.path.dirname(os.path.abspath(__file__))


# ── seed control ─────────────────────────────────────────────────────────────
def set_seed(seed: int) -> torch.Generator:
    """
    Set all randomness sources and return a seeded Generator for DataLoaders.
    Covers: model init weights, dropout mask, Adam state, shuffle order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── one full train + eval run ─────────────────────────────────────────────────
def run_one(feature_source: str, seed: int) -> dict:
    """
    Train from scratch and evaluate on test set.

    Args:
        feature_source: 'vgg19'    → AVEDatasetH5 (h5 files)
                        'r2plus1d' → AVEDataset(.pt files, use_preextracted=True)
        seed: integer random seed

    Returns:
        dict with accuracy, recall, mean_iou, iou05, epochs_run, train_min
    """
    tag = f"{feature_source}_seed{seed}"
    log.info("")
    log.info("=" * 60)
    log.info(f"RUN  feature={feature_source}  seed={seed}")
    log.info("=" * 60)

    g = set_seed(seed)

    # ── datasets ──
    if feature_source == "vgg19":
        assert os.path.exists(config.AUDIO_H5_FILE) and os.path.exists(config.VISUAL_H5_FILE), \
            "h5 files missing — place audio_feature.h5 and visual_feature.h5 in data/"
        train_ds = AVEDatasetH5("train")
        val_ds   = AVEDatasetH5("val")
        test_ds  = AVEDatasetH5("test")
    else:
        train_ds = AVEDataset("train", use_preextracted=True)
        val_ds   = AVEDataset("val",   use_preextracted=True)
        test_ds  = AVEDataset("test",  use_preextracted=True)

    # Seeded generator only for the shuffle (train loader); val/test are not shuffled.
    tr_ldr = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        drop_last=True, generator=g,
    )
    va_ldr = DataLoader(val_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    te_ldr = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    log.info(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    # ── model, loss, optimizer ──
    cw        = utils.compute_class_weights(utils.load_split_file(config.TRAIN_SET_FILE)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    model     = AVEModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_loss  = float("inf")
    no_improve = 0
    best_ckpt  = os.path.join(config.CHECKPOINT_DIR, f"best_model_{tag}.pt")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    t0 = time.time()

    # ── training loop ──
    epochs_run = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epochs_run = epoch

        # train
        model.train()
        tl, tp, tlab = 0.0, [], []
        for b in tr_ldr:
            a = b["audio"].to(DEVICE); v = b["video"].to(DEVICE); l = b["labels"].to(DEVICE)
            optimizer.zero_grad()
            out  = model(a, v, modality_dropout_prob=config.MODALITY_DROPOUT_PROB)
            loss = criterion(out.reshape(-1, config.NUM_CLASSES), l.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tl += loss.item(); tp.append(out.argmax(-1).cpu()); tlab.append(l.cpu())
        tp   = torch.cat([x.reshape(-1) for x in tp])
        tlab = torch.cat([x.reshape(-1) for x in tlab])
        tr_loss = tl / len(tr_ldr)
        tr_acc  = ev.per_second_accuracy(tp, tlab)
        tr_rec  = ev.per_second_recall(tp, tlab)

        # val
        model.eval()
        vl, vp, vlab = 0.0, [], []
        with torch.no_grad():
            for b in va_ldr:
                a = b["audio"].to(DEVICE); v = b["video"].to(DEVICE); l = b["labels"].to(DEVICE)
                out  = model(a, v, modality_dropout_prob=0.0)
                loss = criterion(out.reshape(-1, config.NUM_CLASSES), l.reshape(-1))
                vl += loss.item(); vp.append(out.argmax(-1).cpu()); vlab.append(l.cpu())
        vp   = torch.cat([x.reshape(-1) for x in vp])
        vlab = torch.cat([x.reshape(-1) for x in vlab])
        va_loss = vl / len(va_ldr)
        va_acc  = ev.per_second_accuracy(vp, vlab)
        va_rec  = ev.per_second_recall(vp, vlab)

        scheduler.step(va_loss)
        elapsed = (time.time() - t0) / 60

        log.info(
            f"  Ep {epoch:3d}/{config.NUM_EPOCHS} | "
            f"tr loss {tr_loss:.4f} acc {tr_acc:.3f} rec {tr_rec:.3f} | "
            f"va loss {va_loss:.4f} acc {va_acc:.3f} rec {va_rec:.3f} | "
            f"{elapsed:.0f} min"
        )

        if va_loss < best_loss:
            best_loss  = va_loss; no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            log.info(f"    -> checkpoint saved (val_loss={best_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOPPING_PATIENCE:
                log.info(f"  Early stopping at epoch {epoch}.")
                break

    train_min = (time.time() - t0) / 60
    log.info(f"Training done: {epochs_run} epochs  {train_min:.1f} min  best_val_loss={best_loss:.4f}")

    # ── evaluation ──
    model.load_state_dict(torch.load(best_ckpt, weights_only=True, map_location=DEVICE))
    model.eval()

    pf, lf, pc, lc = [], [], [], []
    with torch.no_grad():
        for b in te_ldr:
            a = b["audio"].to(DEVICE); v = b["video"].to(DEVICE); labs = b["labels"]
            out = model(a, v, modality_dropout_prob=0.0).argmax(-1).cpu()
            for p, l in zip(out.numpy(), labs.numpy()):
                pc.append(p); lc.append(l)
                pf.extend(p.tolist()); lf.extend(l.tolist())

    pt = torch.tensor(pf, dtype=torch.long)
    lt = torch.tensor(lf, dtype=torch.long)
    acc  = ev.per_second_accuracy(pt, lt)
    rec  = ev.per_second_recall(pt, lt)
    ious = np.array([ev.temporal_iou_single(p, g_) for p, g_ in zip(pc, lc)])
    miou = float(ious.mean())
    iou5 = float((ious >= 0.5).mean())

    log.info(f"  Test: acc={acc:.4f}  recall={rec:.4f}  mIoU={miou:.4f}  IoU>=0.5={iou5:.4f}")

    # ── save per-run results file ──
    report = classification_report(
        lf, pf,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.ALL_CLASSES,
        zero_division=0,
    )
    out_path = os.path.join(ROOT, f"results_{tag}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"feature_source : {feature_source}\n")
        f.write(f"seed           : {seed}\n")
        f.write(f"epochs_run     : {epochs_run}\n")
        f.write(f"train_min      : {train_min:.1f}\n\n")
        f.write(f"Per-second accuracy    : {acc:.4f}\n")
        f.write(f"Macro recall           : {rec:.4f}\n")
        f.write(f"Mean Temporal IoU      : {miou:.4f}\n")
        f.write(f"Clips with IoU >= 0.5  : {iou5:.4f}\n\n")
        f.write(report)
    log.info(f"  Saved -> {out_path}")

    # Clean up per-run checkpoint to save disk space (keep only summary)
    if os.path.exists(best_ckpt):
        os.remove(best_ckpt)

    return {"feature": feature_source, "seed": seed,
            "acc": acc, "recall": rec, "miou": miou, "iou05": iou5,
            "epochs": epochs_run, "train_min": train_min}


# ── summary table ─────────────────────────────────────────────────────────────
def write_summary(results: list[dict]) -> None:
    vgg   = [r for r in results if r["feature"] == "vgg19"]
    r2p1d = [r for r in results if r["feature"] == "r2plus1d"]

    def stats(rows, key):
        vals = [r[key] for r in rows]
        return np.mean(vals), np.std(vals, ddof=1 if len(vals) > 1 else 0)

    lines = ["# Multi-Seed Comparison: VGG19 h5 vs R(2+1)D .pt Features\n"]
    lines.append(f"Seeds: {SEEDS}   |   Runs per source: {len(SEEDS)}   |   "
                 f"Total training runs: {len(results)}\n")

    lines.append("\n## Per-Run Results\n")
    lines.append("| Feature source | Seed | Accuracy | Macro Recall | Mean IoU | IoU≥0.5 | Epochs | Time (min) |")
    lines.append("|----------------|------|----------|--------------|----------|---------|--------|------------|")
    for r in sorted(results, key=lambda x: (x["feature"], x["seed"])):
        src = "VGG19 h5" if r["feature"] == "vgg19" else "R(2+1)D .pt"
        lines.append(
            f"| {src} | {r['seed']} "
            f"| {r['acc']:.4f} | {r['recall']:.4f} "
            f"| {r['miou']:.4f} | {r['iou05']:.4f} "
            f"| {r['epochs']} | {r['train_min']:.1f} |"
        )

    lines.append("\n## Mean ± Std (across 3 seeds)\n")
    lines.append("| Feature source | Accuracy | Macro Recall | Mean IoU | IoU≥0.5 |")
    lines.append("|----------------|----------|--------------|----------|---------|")
    for label, rows in [("VGG19 h5", vgg), ("R(2+1)D .pt", r2p1d)]:
        a_m, a_s = stats(rows, "acc")
        r_m, r_s = stats(rows, "recall")
        i_m, i_s = stats(rows, "miou")
        f_m, f_s = stats(rows, "iou05")
        lines.append(
            f"| {label} "
            f"| {a_m:.4f} ± {a_s:.4f} "
            f"| {r_m:.4f} ± {r_s:.4f} "
            f"| {i_m:.4f} ± {i_s:.4f} "
            f"| {f_m:.4f} ± {f_s:.4f} |"
        )

    # Verdict
    a_vgg_m,  a_vgg_s  = stats(vgg,   "acc")
    a_r2_m,   a_r2_s   = stats(r2p1d, "acc")
    rc_vgg_m, rc_vgg_s = stats(vgg,   "recall")
    rc_r2_m,  rc_r2_s  = stats(r2p1d, "recall")

    acc_gap    = abs(a_vgg_m  - a_r2_m)
    recall_gap = abs(rc_vgg_m - rc_r2_m)
    acc_spread = a_vgg_s  + a_r2_s
    rec_spread = rc_vgg_s + rc_r2_s

    lines.append("\n## Verdict\n")
    # Is one clearly better on both primary metrics, with gap > combined std?
    vgg_wins_acc    = a_vgg_m  > a_r2_m  and acc_gap    > acc_spread
    r2p1d_wins_acc  = a_r2_m   > a_vgg_m and acc_gap    > acc_spread
    vgg_wins_recall = rc_vgg_m > rc_r2_m and recall_gap > rec_spread
    r2p1d_wins_rec  = rc_r2_m  > rc_vgg_m and recall_gap > rec_spread

    if vgg_wins_acc and vgg_wins_recall:
        lines.append(
            f"**VGG19 h5 features are the better feature source** for this model and task. "
            f"VGG19 leads by {acc_gap:.1%} accuracy and {recall_gap:.1%} macro recall, "
            f"both exceeding the combined spread ({acc_spread:.1%} / {rec_spread:.1%}), "
            f"so the difference is not explained by run-to-run variance."
        )
    elif r2p1d_wins_acc and r2p1d_wins_rec:
        lines.append(
            f"**R(2+1)D .pt features are the better feature source** for this model and task. "
            f"R(2+1)D leads by {acc_gap:.1%} accuracy and {recall_gap:.1%} macro recall, "
            f"both exceeding the combined spread ({acc_spread:.1%} / {rec_spread:.1%})."
        )
    else:
        lines.append(
            f"**No statistically clear winner from {len(SEEDS)} seeds each.** "
            f"The accuracy gap ({acc_gap:.1%}) vs combined std ({acc_spread:.1%}) and "
            f"the recall gap ({recall_gap:.1%}) vs combined std ({rec_spread:.1%}) "
            f"indicate the spreads overlap substantially — more runs would be needed "
            f"for a confident conclusion."
        )

    total_min = sum(r["train_min"] for r in results)
    lines.append(f"\n**Total compute time (all {len(results)} runs):** {total_min:.1f} min "
                 f"({total_min/60:.1f} h)")

    out_path = os.path.join(ROOT, "results_seed_comparison.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"\nSummary saved -> {out_path}")
    log.info("\n" + "\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not torch.cuda.is_available():
        log.warning("No CUDA GPU detected — runs will be slow on CPU.")
    else:
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}  ({props.total_memory // 1024**2} MB)")

    # Verify prerequisites
    h5_ok = os.path.exists(config.AUDIO_H5_FILE) and os.path.exists(config.VISUAL_H5_FILE)
    pt_ok = (os.path.isdir(os.path.join(config.FEATURES_DIR, "audio")) and
             len(os.listdir(os.path.join(config.FEATURES_DIR, "audio"))) > 0)

    log.info(f"h5  features present : {h5_ok}  (required for vgg19 runs)")
    log.info(f".pt features present : {pt_ok}  (required for r2plus1d runs)")
    if not h5_ok:
        log.error("audio_feature.h5 / visual_feature.h5 missing from data/ — aborting.")
        sys.exit(1)
    if not pt_ok:
        log.error("No .pt files found in data/features/audio/ — aborting.")
        sys.exit(1)

    t_total = time.time()
    results = []

    # 3 × VGG19, then 3 × R(2+1)D
    for src in ["vgg19", "r2plus1d"]:
        for seed in SEEDS:
            results.append(run_one(src, seed))

    log.info(f"\nAll {len(results)} runs complete. Total: {(time.time()-t_total)/60:.1f} min")
    write_summary(results)
