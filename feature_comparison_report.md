# Feature Comparison Report: Yadukrishnan vs Pallavi

Generated: 2026-06-25 10:03

## Environments

| | Yadukrishnan | Pallavi |
|---|---|---|
| PyTorch | 2.6.0+cu124 | 2.12.1 |
| torchvision | 0.21.0+cu124 | 0.27.1 |
| Device | CUDA (RTX 4050) | MPS (Apple Silicon) |
| Video encoder | R(2+1)D-18 (Kinetics) | R(2+1)D-18 (Kinetics) |
| Audio encoder | VGGish | VGGish |

> **Note:** My video files used here are from the R(2+1)D extraction backup (`data/features_r2plus1d_backup/video/`), matching Pallavi's encoder. My current `data/features/video/` contains the newer VGG19 extraction.

## Per-File Comparison

| video_id | modality | max_diff | mean_diff | my_mean_mag | her_mean_mag | mean_diff% | allclose(1e-4) |
|---|---|---|---|---|---|---|---|
| `--5zANFBYzQ` | audio | 2.741136 | 0.167104 | 0.0000 | 0.1671 | N/A (zeros) | False |
| `--5zANFBYzQ` | video | 0.916327 | 0.059299 | 1.0731 | 1.0673 | 5.53% | False |
| `--FJhcfXeow` | audio | 1.377548 | 0.121000 | 0.0000 | 0.1210 | N/A (zeros) | False |
| `--FJhcfXeow` | video | 0.812312 | 0.035966 | 1.1267 | 1.1232 | 3.19% | False |
| `--12UOziMF0` | audio | 3.156848 | 0.133093 | 0.0000 | 0.1331 | N/A (zeros) | False |
| `--12UOziMF0` | video | 0.769954 | 0.041373 | 1.0639 | 1.0601 | 3.89% | False |
| `---1_cCGK4M` | audio | 2.997370 | 0.140026 | 0.0000 | 0.1400 | N/A (zeros) | False |
| `---1_cCGK4M` | video | 0.645989 | 0.034432 | 0.9867 | 0.9869 | 3.49% | False |
| `--9O4XZOge4` | audio | 1.457129 | 0.120182 | 0.0000 | 0.1202 | N/A (zeros) | False |
| `--9O4XZOge4` | video | 1.311972 | 0.077240 | 0.8796 | 0.8786 | 8.78% | False |

## Verdict

AUDIO: My audio features are all-zero tensors. This is caused by librosa failing to decode mp4 audio on Windows (missing ffmpeg backend) — the zeros fallback triggered for all 5 videos. Pallavi's audio features are real VGGish embeddings (mean magnitude ~0.1363). This is a critical extraction bug on my side, not a version difference.

VIDEO (R(2+1)D): Features differ by 4.98% on average — moderate difference, possibly attributable to CUDA vs MPS numerical precision.

