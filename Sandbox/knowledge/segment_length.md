# Segment Length

## Finding

seg=600 (20 seconds at 30fps) is the optimal segment length for GBR-based SpO2 prediction.

## Evidence (v3 experiments)

| seg | segments | subjects | overall R2 |
|-----|----------|----------|------------|
| 300 | 4859 | 246 | 0.070 |
| 450 | 3128 | 216 | 0.124 |
| **600** | **2275** | **183** | **0.215** |
| 900 | 1424 | 134 | 0.208 |

## Why seg=600 works

1. **Enough temporal info per segment**: AC/DC ratio statistics stabilize over ~20s windows
2. **Enough segments**: GBR needs sufficient data to generalize (2275 > minimum threshold)
3. **Enough subjects pass filter**: MIN_SpO2_RANGE=10 filter retains 183 subjects
4. **Not too noisy**: seg=300 includes more subjects but each segment's aggregated features are noisier

## Pitfall

- Shorter segments (300/450) seem better because they include more subjects, but the increased noise overwhelms GBR
- In v7 experiments, seg=300 with device indicators actually made prc2 datasets worse (i16 R2 dropped from 0.134 to -0.220)
- seg=900 loses too many subjects (134 vs 183) due to the SpO2 range filter

## Aggregation statistics

4 stats per raw feature: mean, std, p10, p90. Tested adding skewness/kurtosis (v4-era) but no improvement. 19 raw features x 4 stats = 76 features is the standard set.

Feature selection (v3 Exp 2) showed top-15/25/40 features didn't improve over all 76 — segment length matters far more than feature count.
