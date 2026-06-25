# Audio-Visual Event Localization (AVE)

Per-second audio-visual event localization on the AVE dataset using a BiLSTM attention network.  
Given a 10-second video clip, the model predicts the event category for every second — 28 event categories + Background.

> Based on: **Tian et al., "Audio-Visual Event Localization in Unconstrained Videos", ECCV 2018**

---

## Results

All numbers are **mean ± std across 3 seeds** (42, 123, 2024) unless noted.

| Feature source | Audio | Accuracy | Macro Recall | Mean IoU | IoU ≥ 0.5 |
|----------------|-------|----------|--------------|----------|------------|
| **VGG19 self-extracted** | **real VGGish** | **70.00% ± 0.39%** | **82.72% ± 1.49%** | 80.17% ± 1.17% | 82.01% ± 1.69% |
| R(2+1)D self-extracted | real VGGish | 68.83% ± 1.22% | 80.77% ± 2.00% | 80.41% ± 0.64% | 81.76% ± 0.76% |
| Official h5 (Tian et al.) | VGGish h5 | 66.98% ± 1.14% | 78.19% ± 1.08% | 80.09% ± 0.46% | 82.34% ± 0.25% |
| VGG19 self-extracted | zero (bug) ¹ | 61.27% (n=1) | 73.97% (n=1) | 80.40% | 82.09% |
| R(2+1)D self-extracted | zero (bug) ¹ | 58.45% ± 1.38% | 71.15% ± 1.00% | 81.35% ± 0.08% | 82.92% ± 0.28% |

¹ *Audio was all-zero tensors due to a librosa/ffmpeg PATH bug on Windows — fixed 2026-06-25.*

**Best result: VGG19 self-extracted + real audio (70.00% acc / 82.72% recall) exceeds the official h5 baseline by +3.0 pp accuracy and +4.5 pp recall.**  
Full per-class breakdown: [`results_vgg19_real_audio.txt`](results_vgg19_real_audio.txt) · [`results_r2plus1d_real_audio.txt`](results_r2plus1d_real_audio.txt)

---

## Feature Sources

Two feature sources have been fully evaluated:

### 1. Official pre-extracted h5 (paper reference)
Released by Tian et al. alongside [YapengTian/AVE-ECCV18](https://github.com/YapengTian/AVE-ECCV18):
- **Audio**: VGGish embeddings — `data/audio_feature.h5` (42 MB, shape 4143 × 10 × 128)
- **Video**: VGG19 pool5 — `data/visual_feature.h5` (7.7 GB, shape 4143 × 10 × 7 × 7 × 512)

When both h5 files are present in `data/`, the pipeline uses `AVEDatasetH5` automatically.

### 2. Self-extracted .pt features (team exercise)
Extracted from raw `.mp4` files using `feature_extractor.py`:
- **Audio**: VGGish via ffmpeg subprocess → `data/features/audio/<id>.pt` (10 × 128)
- **Video (primary)**: VGG19 pool5 (ImageNet weights, `model.features`) → `data/features/video/<id>.pt` (10 × 49 × 512)
- **Video (archived)**: R(2+1)D-18 → `data/features_r2plus1d_backup/video/<id>.pt` (10 × 49 × 512)

4,097 clips extracted (excludes 46 clips missing mp4 files).

> **Audio extraction note**: The original `librosa.load()` call silently produced all-zero tensors on Windows
> because librosa's audioread backend could not find ffmpeg in the conda env PATH. This was fixed by replacing
> librosa with a direct `ffmpeg` subprocess call (`pcm_s16le` pipe). All 4,097 audio files were re-extracted
> and verified non-zero before the final retraining runs.

---

## Dataset

- **AVE dataset**: 4,143 video clips × 10 seconds, 28 event categories + Background = 29 classes
- Download videos: [Google Drive](https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK)
- Download pre-extracted features (7.7 GB): released alongside the paper
- Annotations: `Category & VideoID & Quality & StartTime & EndTime` per line
- Splits: train (3,339) / val (402) / test (402)

### Folder structure

```
Audio_visual_event_localization/
├── data/
│   ├── audio_feature.h5              ← official VGGish features (42 MB)   [h5 runs]
│   ├── visual_feature.h5             ← official VGG19 features (7.7 GB)   [h5 runs]
│   ├── Annotations.txt
│   ├── trainSet.txt / valSet.txt / testSet.txt
│   ├── AVE/                          ← raw .mp4 files (needed for self-extraction)
│   ├── features/
│   │   ├── audio/                    ← self-extracted VGGish .pt (~25 MB, 4097 files)
│   │   └── video/                    ← self-extracted VGG19 pool5 .pt (~3.9 GB)
│   └── features_r2plus1d_backup/
│       └── video/                    ← archived R(2+1)D .pt features (~3.9 GB)
├── checkpoints/
└── ...
```

---

## Architecture

```
Audio (10 × 128)                Video (10 × 7 × 7 × 512)
[VGGish features]               [VGG19 pool5 features]
       │                                    │
       │                         reshape → (10, 49, 512)
       │                                    │
       └──────── AudioGuidedAttention ──────┘
                         │
                attended video (10, 512)
                         │
         ┌───────────────┴───────────────┐
    BiLSTM (audio)                  BiLSTM (video)
    128 → 256                       512 → 512
         │                              │
         └──────────── concat ──────────┘
                         │
                    768 → fc(256) → fc(29)
                         │
                per-second logits (10, 29)
```

**Key design choices**
- **Audio-guided attention**: audio embedding selects relevant spatial regions from 49 video patches
- **Modality dropout** (p=0.1): randomly zeros one modality per batch item to prevent dominance
- **Class-weighted CrossEntropyLoss**: inverse-frequency weights handle 29-class imbalance
- **Gradient clipping** at max norm 5.0 for BiLSTM stability
- 2,111,005 trainable parameters

---

## Project Structure

```
Audio_visual_event_localization/
├── config.py                    # All hyperparameters, paths, and constants
├── utils.py                     # Annotation parsing, label building, class weights
├── dataset.py                   # AVEDataset (.pt) + AVEDatasetH5 (h5)
├── models.py                    # AVEModel, AudioGuidedAttention
├── feature_extractor.py         # VGGish + VGG19 pool5 self-extraction
├── evaluate.py                  # Accuracy, recall, temporal IoU metrics
├── run_pipeline.py              # End-to-end: extract → train → evaluate (h5 path)
├── run_multiseed.py             # 3-seed comparison: h5 vs self-extracted
├── run_real_audio.py            # 3-seed retraining after audio bug fix
├── run_audio_extraction.py      # Standalone audio-only re-extraction
│
├── results_vgg19_real_audio.txt            ← best: VGG19 + real audio (3-seed agg)
├── results_r2plus1d_real_audio.txt         ← R(2+1)D + real audio (3-seed agg)
├── results_real_audio_comparison.md        ← before/after audio-fix table
├── results_seed_comparison.md             ← h5 vs R(2+1)D zero-audio comparison
├── results_vgg19_seed{42,123,2024}.txt    ← h5 per-run detail
├── results_r2plus1d_real_audio_seed{42,123,2024}.txt
├── results_vgg19_real_audio_seed{42,123,2024}.txt
└── audio_extraction_failures.txt          ← "no failures" (4097/4097 succeeded)
```

---

## Setup

```bash
conda create -n torch_env python=3.10
conda activate torch_env

# PyTorch with CUDA 12.4 (adjust for your CUDA version)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

pip install numpy h5py opencv-python scikit-learn

# ffmpeg required for self-extraction audio pipeline (librosa replaced by subprocess)
conda install -c conda-forge "ffmpeg=6.*"
```

**Requirements**: NVIDIA GPU with CUDA. For h5 runs: place `audio_feature.h5` and `visual_feature.h5` in `data/`. For self-extraction runs: place `.mp4` files in `data/AVE/`.

---

## Run

### Option A — Official h5 features (paper comparison)
```bash
python run_pipeline.py
```
Requires `data/audio_feature.h5` and `data/visual_feature.h5`. Skips extraction automatically.

### Option B — Self-extract then train
```bash
# Step 1: extract features (~12 min audio + ~8–9 hr video on RTX 4050)
python feature_extractor.py

# Step 2: train with self-extracted features (3 seeds × 2 backbones)
python run_real_audio.py
```

### Multi-seed comparison
```bash
python run_multiseed.py   # 3 seeds × h5 vs .pt, saves results_seed_comparison.md
```

| Stage | Description | Time (RTX 4050) |
|-------|-------------|-----------------|
| Audio extraction | 4,097 videos via VGGish | ~12 min |
| Video extraction | 4,097 videos via VGG19 pool5 | ~8–9 hr |
| Training (one seed) | 30 epochs max, early stop patience=5 | ~3–6 min |

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Modality dropout | 0.1 |
| Gradient clip | max norm 5.0 |

---

## Metrics

- **Macro recall** — average recall across 28 event classes, Background excluded *(primary metric)*
- **Per-second accuracy** — fraction of seconds correctly classified
- **Temporal IoU** — intersection-over-union of predicted vs ground-truth event windows per clip

### Baseline comparison

| Baseline | Accuracy | Macro Recall |
|----------|----------|--------------|
| Always predict Background | ~23% | 0.0% |
| Random uniform (29 classes) | ~3.4% | ~3.4% |
| R(2+1)D + zero audio (buggy, 3-seed) | 58.5% | 71.2% |
| Official h5 / Tian et al. (3-seed) | 67.0% | 78.2% |
| R(2+1)D + real audio (3-seed) | 68.8% | 80.8% |
| **VGG19 self-extracted + real audio (3-seed)** | **70.0%** | **82.7%** |

---

## Citation

```bibtex
@inproceedings{TianECCV2018,
    title={Audio-Visual Event Localization in Unconstrained Videos},
    author={Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
    booktitle={Computer Vision -- ECCV 2018},
    year={2018},
    publisher={Springer}
}
```

---

## Hardware

Tested on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.4 / driver 591.74, Windows 11.
