"""
Retrain with real audio features (4097/4097 non-zero VGGish, fixed 2026-06-25).
Two variants × 3 seeds = 6 runs:

  Variant A  r2plus1d_real_audio : R(2+1)D video (backup dir) + real audio
  Variant B  vgg19_real_audio    : VGG19 pool5 video (current dir) + real audio

Outputs (project root, never overwriting old result files):
  results_r2plus1d_real_audio_seed{S}.txt   per-run detail
  results_vgg19_real_audio_seed{S}.txt      per-run detail
  results_r2plus1d_real_audio.txt           R(2+1)D 3-seed aggregate
  results_vgg19_real_audio.txt              VGG19   3-seed aggregate
  results_real_audio_comparison.md          before/after table + verdict
  real_audio_training.log                   epoch-level log
"""

import os, sys, time, logging, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, utils
from models import AVEModel
from dataset import AVEDataset
import evaluate as ev

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
R2P1D_VID  = os.path.join(ROOT, "data", "features_r2plus1d_backup", "video")
VGG19_VID  = os.path.join(ROOT, "data", "features", "video")
AUDIO_DIR  = os.path.join(ROOT, "data", "features", "audio")

SEEDS   = [42, 123, 2024]
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(ROOT, "real_audio_training.log")
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


# ── dataset subclass with configurable feature dirs ───────────────────────────
class AVEDatasetPT(AVEDataset):
    """AVEDataset with explicit audio_dir and video_dir overrides."""

    def __init__(self, split: str, video_dir: str, audio_dir: str):
        super().__init__(split, use_preextracted=True)
        self._audio_dir = audio_dir
        self._video_dir = video_dir

    def _load_preextracted(self, video_id):
        ap = os.path.join(self._audio_dir, f"{video_id}.pt")
        vp = os.path.join(self._video_dir, f"{video_id}.pt")
        if os.path.exists(ap) and os.path.exists(vp):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = torch.load(ap, weights_only=True)
                video = torch.load(vp, weights_only=True)
            return audio, video
        warnings.warn(f"Features missing for {video_id} — falling back to raw")
        return self._load_raw(video_id)


# ── seed control ──────────────────────────────────────────────────────────────
def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── one train + eval run ──────────────────────────────────────────────────────
def run_one(variant: str, seed: int, video_dir: str) -> dict:
    tag = f"{variant}_seed{seed}"
    log.info("")
    log.info("=" * 60)
    log.info(f"RUN  variant={variant}  seed={seed}")
    log.info("=" * 60)

    g = set_seed(seed)

    train_ds = AVEDatasetPT("train", video_dir=video_dir, audio_dir=AUDIO_DIR)
    val_ds   = AVEDatasetPT("val",   video_dir=video_dir, audio_dir=AUDIO_DIR)
    test_ds  = AVEDatasetPT("test",  video_dir=video_dir, audio_dir=AUDIO_DIR)

    tr_ldr = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        drop_last=True, generator=g,
    )
    va_ldr = DataLoader(val_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    te_ldr = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    log.info(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    cw        = utils.compute_class_weights(utils.load_split_file(config.TRAIN_SET_FILE)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    model     = AVEModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_loss  = float("inf")
    no_improve = 0
    best_ckpt  = os.path.join(config.CHECKPOINT_DIR, f"best_{tag}.pt")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    t0 = time.time()

    epochs_run = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epochs_run = epoch

        # ── train ──
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
        tr_acc = ev.per_second_accuracy(tp, tlab)
        tr_rec = ev.per_second_recall(tp, tlab)

        # ── val ──
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
            f"tr acc {tr_acc:.3f} rec {tr_rec:.3f} | "
            f"va loss {va_loss:.4f} acc {va_acc:.3f} rec {va_rec:.3f} | "
            f"{elapsed:.0f} min"
        )

        if va_loss < best_loss:
            best_loss = va_loss; no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            log.info(f"    -> checkpoint (val_loss={best_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOPPING_PATIENCE:
                log.info(f"  Early stop at epoch {epoch}.")
                break

    train_min = (time.time() - t0) / 60
    log.info(f"Training done: {epochs_run} epochs  {train_min:.1f} min")

    # ── test evaluation ──
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

    # ── per-run file ──
    report = classification_report(
        lf, pf,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.ALL_CLASSES,
        zero_division=0,
    )
    out_path = os.path.join(ROOT, f"results_{tag}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"variant        : {variant}\n")
        f.write(f"seed           : {seed}\n")
        f.write(f"epochs_run     : {epochs_run}\n")
        f.write(f"train_min      : {train_min:.1f}\n")
        f.write(f"audio_dir      : {AUDIO_DIR}\n")
        f.write(f"video_dir      : {video_dir}\n\n")
        f.write(f"Per-second accuracy    : {acc:.4f}\n")
        f.write(f"Macro recall           : {rec:.4f}\n")
        f.write(f"Mean Temporal IoU      : {miou:.4f}\n")
        f.write(f"Clips with IoU >= 0.5  : {iou5:.4f}\n\n")
        f.write(report)
    log.info(f"  Saved -> {out_path}")

    if os.path.exists(best_ckpt):
        os.remove(best_ckpt)

    return {"variant": variant, "seed": seed,
            "acc": acc, "recall": rec, "miou": miou, "iou05": iou5,
            "epochs": epochs_run, "train_min": train_min}


# ── aggregate per-variant file ────────────────────────────────────────────────
def write_aggregate(variant: str, rows: list[dict]) -> dict:
    """Write a 3-seed aggregate .txt and return mean/std dict."""
    accs   = [r["acc"]    for r in rows]
    recs   = [r["recall"] for r in rows]
    mious  = [r["miou"]   for r in rows]
    iou5s  = [r["iou05"]  for r in rows]

    ddof = 1 if len(rows) > 1 else 0
    stats = {
        "acc_mean":  np.mean(accs),  "acc_std":  np.std(accs,  ddof=ddof),
        "rec_mean":  np.mean(recs),  "rec_std":  np.std(recs,  ddof=ddof),
        "miou_mean": np.mean(mious), "miou_std": np.std(mious, ddof=ddof),
        "iou5_mean": np.mean(iou5s), "iou5_std": np.std(iou5s, ddof=ddof),
    }

    out_path = os.path.join(ROOT, f"results_{variant}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"variant : {variant}\n")
        f.write(f"seeds   : {[r['seed'] for r in rows]}\n\n")
        f.write("Per-run:\n")
        for r in rows:
            f.write(f"  seed {r['seed']:4d}  acc={r['acc']:.4f}  recall={r['recall']:.4f}"
                    f"  mIoU={r['miou']:.4f}  IoU>=0.5={r['iou05']:.4f}"
                    f"  ({r['epochs']} ep, {r['train_min']:.0f} min)\n")
        f.write(f"\nMean ± std (n={len(rows)}):\n")
        f.write(f"  Accuracy   : {stats['acc_mean']:.4f} ± {stats['acc_std']:.4f}\n")
        f.write(f"  Recall     : {stats['rec_mean']:.4f} ± {stats['rec_std']:.4f}\n")
        f.write(f"  Mean IoU   : {stats['miou_mean']:.4f} ± {stats['miou_std']:.4f}\n")
        f.write(f"  IoU >= 0.5 : {stats['iou5_mean']:.4f} ± {stats['iou5_std']:.4f}\n")
    log.info(f"Aggregate saved -> {out_path}")
    return stats


# ── comparison markdown ───────────────────────────────────────────────────────
def parse_old_results(paths: list[str]) -> list[dict] | None:
    """Parse acc/recall/mIoU/IoU05 from old per-run .txt files."""
    rows = []
    for p in paths:
        if not os.path.exists(p):
            return None
        d = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                if "Per-second accuracy" in line:
                    d["acc"] = float(line.split(":")[-1].strip())
                elif "Macro recall" in line:
                    d["recall"] = float(line.split(":")[-1].strip())
                elif "Mean Temporal IoU" in line:
                    d["miou"] = float(line.split(":")[-1].strip())
                elif "Clips with IoU >= 0.5" in line:
                    d["iou05"] = float(line.split(":")[-1].strip())
        if len(d) == 4:
            rows.append(d)
    return rows if len(rows) == len(paths) else None


def stats_of(rows: list[dict], key: str):
    vals = [r[key] for r in rows]
    ddof = 1 if len(vals) > 1 else 0
    return np.mean(vals), np.std(vals, ddof=ddof)


def write_comparison(new_r2: dict, new_vgg: dict,
                     old_r2_rows, old_vgg_rows) -> None:
    lines = []
    lines.append("# Real Audio vs Zero-Audio: Before/After Comparison\n")
    lines.append("Audio fix date: 2026-06-25 (ffmpeg subprocess replaces librosa zero-fallback)\n")
    lines.append(f"Seeds: {SEEDS}\n")

    # ── per-run table (new runs) ──
    lines.append("\n## New Runs (Real Audio) — Per-Run\n")
    lines.append("| Variant | Seed | Accuracy | Macro Recall | Mean IoU | IoU≥0.5 | Epochs |")
    lines.append("|---------|------|----------|--------------|----------|---------|--------|")
    for r in sorted(_all_new_rows, key=lambda x: (x["variant"], x["seed"])):
        vname = "R(2+1)D + real audio" if "r2plus1d" in r["variant"] else "VGG19 + real audio"
        lines.append(f"| {vname} | {r['seed']} "
                     f"| {r['acc']:.4f} | {r['recall']:.4f} "
                     f"| {r['miou']:.4f} | {r['iou05']:.4f} | {r['epochs']} |")

    # ── before/after aggregate table ──
    lines.append("\n## Before / After — Mean ± Std (3 seeds each)\n")
    lines.append("| Variant | Audio | Accuracy | Macro Recall | Mean IoU | IoU≥0.5 |")
    lines.append("|---------|-------|----------|--------------|----------|---------|")

    # R(2+1)D rows
    if old_r2_rows:
        a_m, a_s = stats_of(old_r2_rows, "acc")
        r_m, r_s = stats_of(old_r2_rows, "recall")
        i_m, i_s = stats_of(old_r2_rows, "miou")
        f_m, f_s = stats_of(old_r2_rows, "iou05")
        lines.append(f"| R(2+1)D video | **zero (bug)** "
                     f"| {a_m:.4f} ± {a_s:.4f} | {r_m:.4f} ± {r_s:.4f} "
                     f"| {i_m:.4f} ± {i_s:.4f} | {f_m:.4f} ± {f_s:.4f} |")
    else:
        lines.append("| R(2+1)D video | zero (bug) | *old results not found* | | | |")

    lines.append(f"| R(2+1)D video | **real VGGish** "
                 f"| {new_r2['acc_mean']:.4f} ± {new_r2['acc_std']:.4f} "
                 f"| {new_r2['rec_mean']:.4f} ± {new_r2['rec_std']:.4f} "
                 f"| {new_r2['miou_mean']:.4f} ± {new_r2['miou_std']:.4f} "
                 f"| {new_r2['iou5_mean']:.4f} ± {new_r2['iou5_std']:.4f} |")

    # VGG19 rows
    if old_vgg_rows:
        n_old = len(old_vgg_rows)
        suffix = f"(n={n_old})" if n_old < 3 else "(3 seeds)"
        a_m, a_s = stats_of(old_vgg_rows, "acc")
        r_m, r_s = stats_of(old_vgg_rows, "recall")
        i_m, i_s = stats_of(old_vgg_rows, "miou")
        f_m, f_s = stats_of(old_vgg_rows, "iou05")
        lines.append(f"| VGG19 video | **zero (bug)** {suffix} "
                     f"| {a_m:.4f} ± {a_s:.4f} | {r_m:.4f} ± {r_s:.4f} "
                     f"| {i_m:.4f} ± {i_s:.4f} | {f_m:.4f} ± {f_s:.4f} |")
    else:
        lines.append("| VGG19 video | zero (bug) | *old results not found* | | | |")

    lines.append(f"| VGG19 video | **real VGGish** "
                 f"| {new_vgg['acc_mean']:.4f} ± {new_vgg['acc_std']:.4f} "
                 f"| {new_vgg['rec_mean']:.4f} ± {new_vgg['rec_std']:.4f} "
                 f"| {new_vgg['miou_mean']:.4f} ± {new_vgg['miou_std']:.4f} "
                 f"| {new_vgg['iou5_mean']:.4f} ± {new_vgg['iou5_std']:.4f} |")

    # ── audio-fix delta ──
    lines.append("\n## Audio-Fix Delta (real − zero)\n")
    lines.append("| Variant | Δ Accuracy | Δ Macro Recall | Δ Mean IoU |")
    lines.append("|---------|-----------|----------------|------------|")

    if old_r2_rows:
        old_a, _ = stats_of(old_r2_rows, "acc")
        old_r, _ = stats_of(old_r2_rows, "recall")
        old_i, _ = stats_of(old_r2_rows, "miou")
        d_a = new_r2["acc_mean"]  - old_a
        d_r = new_r2["rec_mean"]  - old_r
        d_i = new_r2["miou_mean"] - old_i
        lines.append(f"| R(2+1)D | {d_a:+.4f} ({d_a*100:+.2f}pp) "
                     f"| {d_r:+.4f} ({d_r*100:+.2f}pp) "
                     f"| {d_i:+.4f} ({d_i*100:+.2f}pp) |")

    if old_vgg_rows:
        old_a, _ = stats_of(old_vgg_rows, "acc")
        old_r, _ = stats_of(old_vgg_rows, "recall")
        old_i, _ = stats_of(old_vgg_rows, "miou")
        d_a = new_vgg["acc_mean"]  - old_a
        d_r = new_vgg["rec_mean"]  - old_r
        d_i = new_vgg["miou_mean"] - old_i
        lines.append(f"| VGG19   | {d_a:+.4f} ({d_a*100:+.2f}pp) "
                     f"| {d_r:+.4f} ({d_r*100:+.2f}pp) "
                     f"| {d_i:+.4f} ({d_i*100:+.2f}pp) |")

    # ── reference: official h5 ──
    old_h5_r2 = parse_old_results([
        os.path.join(ROOT, f"results_vgg19_seed{s}.txt") for s in SEEDS
    ])
    if old_h5_r2:
        a_m, a_s = stats_of(old_h5_r2, "acc")
        r_m, r_s = stats_of(old_h5_r2, "recall")
        i_m, i_s = stats_of(old_h5_r2, "miou")
        f_m, f_s = stats_of(old_h5_r2, "iou05")
        lines.append("\n## Reference: Official h5 Features (VGGish + VGG19 pool5 by Tian et al.)\n")
        lines.append("| Accuracy | Macro Recall | Mean IoU | IoU≥0.5 |")
        lines.append("|----------|--------------|----------|---------|")
        lines.append(f"| {a_m:.4f} ± {a_s:.4f} | {r_m:.4f} ± {r_s:.4f} "
                     f"| {i_m:.4f} ± {i_s:.4f} | {f_m:.4f} ± {f_s:.4f} |")
        lines.append(f"\n*(3 seeds: {SEEDS})*\n")

    total_min = sum(r["train_min"] for r in _all_new_rows)
    lines.append(f"\n**Total compute (6 runs):** {total_min:.0f} min ({total_min/60:.1f} h)\n")

    out_path = os.path.join(ROOT, "results_real_audio_comparison.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Comparison saved -> {out_path}")
    log.info("\n" + "\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────
_all_new_rows = []

if __name__ == "__main__":
    if not torch.cuda.is_available():
        log.warning("No CUDA GPU — runs will be slow.")
    else:
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}  ({props.total_memory // 1024**2} MB)")

    # Verify feature dirs
    assert os.path.isdir(R2P1D_VID), f"R(2+1)D backup missing: {R2P1D_VID}"
    assert os.path.isdir(VGG19_VID), f"VGG19 video dir missing: {VGG19_VID}"
    assert os.path.isdir(AUDIO_DIR), f"Audio dir missing: {AUDIO_DIR}"

    r2_count    = len([f for f in os.listdir(R2P1D_VID) if f.endswith(".pt")])
    vgg19_count = len([f for f in os.listdir(VGG19_VID) if f.endswith(".pt")])
    audio_count = len([f for f in os.listdir(AUDIO_DIR) if f.endswith(".pt")])
    log.info(f"R(2+1)D video features : {r2_count}")
    log.info(f"VGG19  video features  : {vgg19_count}")
    log.info(f"Audio features         : {audio_count}")

    t_total = time.time()
    r2_rows, vgg_rows = [], []

    VARIANTS = [
        ("r2plus1d_real_audio", R2P1D_VID),
        ("vgg19_real_audio",    VGG19_VID),
    ]

    for variant, vid_dir in VARIANTS:
        for seed in SEEDS:
            r = run_one(variant, seed, vid_dir)
            _all_new_rows.append(r)
            if "r2plus1d" in variant:
                r2_rows.append(r)
            else:
                vgg_rows.append(r)

    log.info(f"\nAll {len(_all_new_rows)} runs complete. "
             f"Total: {(time.time()-t_total)/60:.1f} min")

    new_r2_stats  = write_aggregate("r2plus1d_real_audio", r2_rows)
    new_vgg_stats = write_aggregate("vgg19_real_audio",    vgg_rows)

    # Load old zero-audio baselines for the comparison table
    old_r2_rows = parse_old_results([
        os.path.join(ROOT, f"results_r2plus1d_seed{s}.txt") for s in SEEDS
    ])
    # VGG19 .pt old: single run with zero audio — stored in results.txt
    # (results_self_extracted_features.txt is the R(2+1)D single-seed baseline, not VGG19 .pt)
    old_vgg_rows = parse_old_results([os.path.join(ROOT, "results.txt")])

    write_comparison(new_r2_stats, new_vgg_stats, old_r2_rows, old_vgg_rows)
