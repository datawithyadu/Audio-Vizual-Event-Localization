# Real Audio vs Zero-Audio: Before/After Comparison

Audio fix date: 2026-06-25 (ffmpeg subprocess replaces librosa zero-fallback)

Seeds: [42, 123, 2024]


## New Runs (Real Audio) — Per-Run

| Variant | Seed | Accuracy | Macro Recall | Mean IoU | IoU≥0.5 | Epochs |
|---------|------|----------|--------------|----------|---------|--------|
| R(2+1)D + real audio | 42 | 0.6744 | 0.7868 | 0.7991 | 0.8109 | 14 |
| R(2+1)D + real audio | 123 | 0.6935 | 0.8267 | 0.8114 | 0.8259 | 13 |
| R(2+1)D + real audio | 2024 | 0.6970 | 0.8095 | 0.8020 | 0.8159 | 13 |
| VGG19 + real audio | 42 | 0.6958 | 0.8133 | 0.7894 | 0.8010 | 16 |
| VGG19 + real audio | 123 | 0.7007 | 0.8429 | 0.8126 | 0.8333 | 12 |
| VGG19 + real audio | 2024 | 0.7035 | 0.8253 | 0.8030 | 0.8259 | 14 |

## Before / After — Mean ± Std (3 seeds each)

| Variant | Audio | Accuracy | Macro Recall | Mean IoU | IoU≥0.5 |
|---------|-------|----------|--------------|----------|---------|
| R(2+1)D video | **zero (bug)** | 0.5845 ± 0.0138 | 0.7115 ± 0.0100 | 0.8135 ± 0.0008 | 0.8292 ± 0.0028 |
| R(2+1)D video | **real VGGish** | 0.6883 ± 0.0122 | 0.8077 ± 0.0200 | 0.8041 ± 0.0064 | 0.8176 ± 0.0076 |
| VGG19 video | **zero (bug)** (n=1) | 0.6127 ± 0.0000 | 0.7397 ± 0.0000 | 0.8040 ± 0.0000 | 0.8209 ± 0.0000 |
| VGG19 video | **real VGGish** | 0.7000 ± 0.0039 | 0.8272 ± 0.0149 | 0.8017 ± 0.0117 | 0.8201 ± 0.0169 |

## Audio-Fix Delta (real − zero)

| Variant | Δ Accuracy | Δ Macro Recall | Δ Mean IoU |
|---------|-----------|----------------|------------|
| R(2+1)D | +0.1038 (+10.38pp) | +0.0962 (+9.62pp) | -0.0094 (-0.94pp) |
| VGG19   | +0.0873 (+8.73pp) | +0.0875 (+8.75pp) | -0.0023 (-0.23pp) |

## Reference: Official h5 Features (VGGish + VGG19 pool5 by Tian et al.)

| Accuracy | Macro Recall | Mean IoU | IoU≥0.5 |
|----------|--------------|----------|---------|
| 0.6698 ± 0.0114 | 0.7819 ± 0.0108 | 0.8009 ± 0.0046 | 0.8234 ± 0.0025 |

*(3 seeds: [42, 123, 2024])*


**Total compute (6 runs):** 22 min (0.4 h)

