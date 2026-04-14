# Best Results Summary

## Target: R2 > 0.1 on all 6 datasets

### Best overall: v13 large_rf (5/6 pass)

```
GBR: Huber, n=200, d=4, lr=0.05
RF:  n=500, d=8, min_leaf=3
Blend: 50/50, device-normalized, median offset
```

| Dataset | R2 | MAE | PCC | Pass |
|---------|-----|-----|-----|------|
| prc-c920 | 0.248 | 3.58 | 0.706 | OK |
| prc-i15 | 0.557 | 2.91 | 0.878 | OK |
| prc-i15m | 0.462 | 3.25 | 0.839 | OK |
| prc2-c930 | 0.007 | 4.14 | 0.703 | X |
| prc2-i16 | 0.207 | 3.74 | 0.772 | OK |
| prc2-i16m | 0.133 | 3.58 | 0.787 | OK |
| **ALL** | **0.270** | **3.54** | **0.781** | **5/6** |

### Best for c930: v12 ensemble (c930=0.013)

```
GBR: Huber, n=200, d=4 + RF: n=200, d=6
50/50 blend, devnorm, median offset
```

c930=0.013, i16m=0.123, overall=0.266

### Best for i16m: v13 large_rf (i16m=0.133)

### Best per-dataset R2 achieved (across all experiments)

| Dataset | Best R2 | Experiment |
|---------|---------|------------|
| c920 | 0.273 | v11 quantile_offset |
| i15 | 0.558 | v11 huber_large_offset |
| i15m | 0.478 | v11 huber_large_offset |
| c930 | 0.014 | v5 devnorm+offset, v11 clip_range |
| i16 | 0.207 | v13 large_rf |
| i16m | 0.133 | v13 large_rf |

### Essential recipe (common to all good configs)

1. seg=600 (20s segments)
2. 19 raw features x 4 agg stats = 76 features
3. Device normalization (z-score per device inside CV)
4. Per-device median offset correction
5. GBR with Huber loss (or GBR+RF ensemble)
