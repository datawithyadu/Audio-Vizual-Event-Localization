# Multi-Seed Comparison: VGG19 h5 vs R(2+1)D .pt Features

Seeds: [42, 123, 2024]   |   Runs per source: 3   |   Total training runs: 6


## Per-Run Results

| Feature source | Seed | Accuracy | Macro Recall | Mean IoU | IoU≥0.5 | Epochs | Time (min) |
|----------------|------|----------|--------------|----------|---------|--------|------------|
| R(2+1)D .pt | 42 | 0.5716 | 0.7046 | 0.8142 | 0.8308 | 11 | 2.6 |
| R(2+1)D .pt | 123 | 0.5828 | 0.7070 | 0.8126 | 0.8308 | 13 | 4.2 |
| R(2+1)D .pt | 2024 | 0.5990 | 0.7230 | 0.8137 | 0.8259 | 13 | 3.5 |
| VGG19 h5 | 42 | 0.6818 | 0.7939 | 0.7966 | 0.8209 | 17 | 8.0 |
| VGG19 h5 | 123 | 0.6684 | 0.7789 | 0.8058 | 0.8259 | 14 | 7.1 |
| VGG19 h5 | 2024 | 0.6592 | 0.7729 | 0.8004 | 0.8234 | 14 | 6.7 |

## Mean ± Std (across 3 seeds)

| Feature source | Accuracy | Macro Recall | Mean IoU | IoU≥0.5 |
|----------------|----------|--------------|----------|---------|
| VGG19 h5 | 0.6698 ± 0.0114 | 0.7819 ± 0.0108 | 0.8009 ± 0.0046 | 0.8234 ± 0.0025 |
| R(2+1)D .pt | 0.5845 ± 0.0138 | 0.7116 ± 0.0100 | 0.8135 ± 0.0008 | 0.8292 ± 0.0029 |

## Verdict

**VGG19 h5 features are the better feature source** for this model and task. VGG19 leads by 8.5% accuracy and 7.0% macro recall, both exceeding the combined spread (2.5% / 2.1%), so the difference is not explained by run-to-run variance.

**Total compute time (all 6 runs):** 32.2 min (0.5 h)
