"""
Pre-extract and save audio (VGGish) and video (VGG19 pool5) features for every
clip in the AVE dataset.  Run once before training:

    python feature_extractor.py

Saved shapes per clip:
    features/audio/<video_id>.pt  →  torch.FloatTensor (10, 128)
    features/video/<video_id>.pt  →  torch.FloatTensor (10, 49, 512)

Audio pipeline  (matches Tian et al. audio_feature_extractor.py)
    MP4 → ffmpeg subprocess (pcm_s16le pipe) → 10 one-second chunks → VGGish → (10, 128)
    Uses ffmpeg directly instead of librosa; librosa's audioread backend is
    unreliable on Windows when ffmpeg is installed inside a conda env (PATH mismatch).

Video pipeline  (matches Tian et al. visual_feature_extractor.py)
    MP4 → 16 frames per second (224×224, RGB) → VGG19 features layer
        → (16, 512, 7, 7) → mean over 16 frames → (512, 7, 7) → reshape (49, 512)
    Produces (10, 49, 512) per clip — identical to h5 block5_pool shape.
"""

import os
import shutil
import subprocess
import warnings
import numpy as np
import torch
import torchvision.models as tvm
import cv2

import config
import utils

# ── VGG19 extraction constants (Tian et al. visual_feature_extractor.py) ─────
_VGG19_FRAME_SIZE     = (224, 224)  # VGG19 canonical input resolution
_VGG19_FRAMES_PER_SEC = 16          # paper's sample_num=16 frames per second

# ImageNet normalisation expected by torchvision VGG19 weights
_VGG19_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_VGG19_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
# VGGish audio encoder
# ──────────────────────────────────────────────

def _find_ffmpeg() -> str:
    """Return path to ffmpeg executable, searching PATH then conda env Library/bin."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        candidate = os.path.join(prefix, "Library", "bin", "ffmpeg.exe")
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "ffmpeg not found. Install with: conda install -c conda-forge 'ffmpeg=6.*'"
    )


def _load_audio_ffmpeg(mp4_path: str, sr: int, duration: float) -> np.ndarray:
    """Decode mp4 audio to float32 mono waveform via ffmpeg subprocess (pcm_s16le pipe)."""
    exe = _find_ffmpeg()
    cmd = [exe, "-v", "quiet", "-i", str(mp4_path),
           "-f", "s16le", "-acodec", "pcm_s16le",
           "-ar", str(sr), "-ac", "1",
           "-t", str(duration),
           "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(
            f"ffmpeg decode failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:200]}"
        )
    wav = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return wav


def build_vggish(device: torch.device):
    """
    Load VGGish from harritaylor/torchvggish via torch.hub.
    postprocess=False: returns raw 128-dim float embeddings (no PCA/quantisation).
    """
    hub_dir = os.path.join(torch.hub.get_dir(), "harritaylor_torchvggish_master")
    if os.path.isdir(hub_dir):
        model = torch.hub.load(hub_dir, "vggish", source="local",
                               postprocess=False, trust_repo=True)
    else:
        model = torch.hub.load("harritaylor/torchvggish", "vggish",
                               postprocess=False, trust_repo=True)
    model.eval()
    model = model.to(device)
    return model


def extract_audio_features(
    mp4_path: str,
    vggish,
    device: torch.device,
    n_segments: int = config.NUM_SEGMENTS,
    target_sr: int = config.AUDIO_SAMPLE_RATE,
) -> torch.Tensor:
    """
    Extract VGGish features for a single video clip.

    Returns:
        FloatTensor (10, 128)
    """
    try:
        y = _load_audio_ffmpeg(mp4_path, target_sr, 10.0)
    except Exception as e:
        warnings.warn(f"Audio load failed for {mp4_path}: {e} — using zeros")
        return torch.zeros(n_segments, config.AUDIO_EMBED_DIM)

    target_len = target_sr * n_segments
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    y = y.astype(np.float32)

    try:
        with torch.no_grad():
            embeddings = vggish.forward(y, target_sr)   # (N, 128)
    except Exception as e:
        warnings.warn(f"VGGish forward failed for {mp4_path}: {e} — using zeros")
        return torch.zeros(n_segments, config.AUDIO_EMBED_DIM)

    embeddings = embeddings.cpu().float()

    if embeddings.size(0) >= n_segments:
        embeddings = embeddings[:n_segments]
    else:
        pad = torch.zeros(n_segments - embeddings.size(0), config.AUDIO_EMBED_DIM)
        embeddings = torch.cat([embeddings, pad], dim=0)

    return embeddings   # (10, 128)


# ──────────────────────────────────────────────
# VGG19 pool5 video encoder  (Tian et al. methodology)
# ──────────────────────────────────────────────

def build_vgg19(device: torch.device) -> torch.nn.Module:
    """
    Load ImageNet-pretrained VGG19 and return only the convolutional feature
    layers (model.features), equivalent to Keras block5_pool output.

    For a 224×224 input, model.features outputs (B, 512, 7, 7) — the same
    spatial shape as block5_pool in the original Keras script.
    """
    weights = tvm.VGG19_Weights.IMAGENET1K_V1
    model   = tvm.vgg19(weights=weights)
    extractor = model.features   # conv layers only; no avgpool or classifier
    extractor.eval()
    return extractor.to(device)


def extract_video_features(
    mp4_path: str,
    extractor: torch.nn.Module,
    device: torch.device,
    n_segments: int = config.NUM_SEGMENTS,
) -> torch.Tensor:
    """
    Extract VGG19 pool5 features for a single video clip.

    Replicates Tian et al. visual_feature_extractor.py:
      - sample_num=16 frames per 1-second segment
      - average over 16 frames → (512, 7, 7) per second
      - reshape to (49, 512)

    All 160 frames (10 sec × 16 frames) are batched into one GPU forward
    pass to minimise launch overhead while keeping the same methodology.

    Returns:
        FloatTensor (10, 49, 512)
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        warnings.warn(f"Cannot open video: {mp4_path} — using zeros")
        return torch.zeros(n_segments, config.VIDEO_NUM_REGIONS, config.VIDEO_FEATURE_DIM)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Move normalisation tensors to device once
    mean = _VGG19_MEAN.to(device)
    std  = _VGG19_STD.to(device)

    seg_features = []
    for seg_idx in range(n_segments):
        start_f = int(seg_idx * fps)
        end_f   = min(int((seg_idx + 1) * fps), total_frames)
        if end_f <= start_f:
            end_f = start_f + 1

        # 16 frame indices uniformly spread across this 1-second window
        indices = np.linspace(start_f, end_f - 1, _VGG19_FRAMES_PER_SEC, dtype=int)
        indices = np.clip(indices, 0, total_frames - 1)

        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, _VGG19_FRAME_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).float() / 255.0  # (H, W, 3)
                frame = frame.permute(2, 0, 1)                   # (3, H, W)
            else:
                frame = torch.zeros(3, *_VGG19_FRAME_SIZE)
            frames.append(frame)

        # Batch all 16 frames → single forward pass through VGG19
        batch = torch.stack(frames, dim=0).to(device)   # (16, 3, 224, 224)
        batch = (batch - mean) / std

        try:
            with torch.no_grad():
                feats = extractor(batch)    # (16, 512, 7, 7)
        except Exception as e:
            warnings.warn(f"VGG19 forward failed for {mp4_path} seg {seg_idx}: {e}")
            feats = torch.zeros(
                _VGG19_FRAMES_PER_SEC, config.VIDEO_FEATURE_DIM,
                config.VIDEO_SPATIAL_SIZE, config.VIDEO_SPATIAL_SIZE,
                device=device,
            )

        # Average over 16 frames — matches np.mean(feature, axis=1) in original
        seg_feat = feats.mean(dim=0).cpu()   # (512, 7, 7)
        seg_feat = seg_feat.permute(1, 2, 0) # (7, 7, 512)
        seg_feat = seg_feat.reshape(config.VIDEO_NUM_REGIONS, config.VIDEO_FEATURE_DIM)
        seg_features.append(seg_feat)

    cap.release()
    return torch.stack(seg_features, dim=0)   # (10, 49, 512)


# ──────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────

def extract_all_features(
    split_files: list[str] | None = None,
    overwrite: bool = False,
    max_videos: int | None = None,
) -> None:
    """
    Pre-extract audio and video features for every unique video in the dataset.

    Args:
        split_files: list of split file paths to process (default: all three)
        overwrite  : re-extract even if the .pt file already exists
        max_videos : stop after this many videos (used for test-batch verification)
    """
    if split_files is None:
        split_files = [config.TRAIN_SET_FILE, config.VAL_SET_FILE, config.TEST_SET_FILE]

    utils.ensure_dirs()
    device = get_device()
    print(f"Using device: {device}")

    print("Loading VGGish …")
    vggish = build_vggish(device)

    print("Loading VGG19 (ImageNet) …")
    extractor = build_vgg19(device)

    # Collect all unique video IDs across all splits
    all_samples = []
    for sf in split_files:
        all_samples.extend(utils.load_split_file(sf))

    seen = set()
    unique_samples = []
    for s in all_samples:
        if s["video_id"] not in seen:
            seen.add(s["video_id"])
            unique_samples.append(s)

    if max_videos is not None:
        unique_samples = unique_samples[:max_videos]

    print(f"Extracting features for {len(unique_samples)} unique videos …\n")

    audio_dir = os.path.join(config.FEATURES_DIR, "audio")
    video_dir = os.path.join(config.FEATURES_DIR, "video")

    for i, sample in enumerate(unique_samples, 1):
        vid      = sample["video_id"]
        mp4_path = utils.get_video_path(vid)

        audio_out = os.path.join(audio_dir, f"{vid}.pt")
        video_out = os.path.join(video_dir, f"{vid}.pt")

        if not os.path.exists(mp4_path):
            print(f"  [{i}/{len(unique_samples)}] SKIP (missing) {vid}")
            continue

        if not overwrite and os.path.exists(audio_out) and os.path.exists(video_out):
            print(f"  [{i}/{len(unique_samples)}] SKIP (exists)  {vid}")
            continue

        print(f"  [{i}/{len(unique_samples)}] {vid}", end=" … ", flush=True)

        if overwrite or not os.path.exists(audio_out):
            af = extract_audio_features(mp4_path, vggish, device)
            torch.save(af, audio_out)

        if overwrite or not os.path.exists(video_out):
            vf = extract_video_features(mp4_path, extractor, device)
            torch.save(vf, video_out)

        print("done")

    print("\nFeature extraction complete.")
    print(f"  Audio saved to : {audio_dir}/")
    print(f"  Video saved to : {video_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite",   action="store_true",
                        help="Re-extract even if .pt files already exist")
    parser.add_argument("--test-batch",  type=int, default=None, metavar="N",
                        help="Only process the first N videos (shape verification)")
    args = parser.parse_args()
    extract_all_features(overwrite=args.overwrite, max_videos=args.test_batch)
