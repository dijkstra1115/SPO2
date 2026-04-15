# PRC2 Recovery Plan

## Goal
Raise every `prc2-*` dataset to test `R2 > 0.1` without assuming all datasets must enter the same model pool.

## Current State
- Latest confirmed `prc2` result from [Sandbox/output/infer_summary_topk20_seg900.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/infer_summary_topk20_seg900.csv):
  - `prc2-c930`: `R2 = -0.1689`
  - `prc2-i16`: `R2 = -0.3832`
  - `prc2-i16m`: `R2 = -0.6605`
- The current `prc2` inference path uses a `prc2`-specialized raw pool with `733` models and `top_k=20`.
- `prc-*` already passes, so `prc2` work should avoid regressing the existing `prc` path.

## What Has Been Ruled Out
- Pool-only tuning is not enough.
  - Prefix-specialized raw pool still leaves all `prc2` datasets below zero R2 in the current summary.
- Simple model substitutions are not enough.
  - Direct LOSO `Ridge` and `HistGradientBoostingRegressor` baselines did not recover `prc2`.
- Local tweaks are not enough.
  - Disabling smoothing, reducing segment length to `450`, allowing same-subject cross-device references, and simple OOF calibration all failed to produce a useful gain.

## Working Hypothesis
- The main bottleneck is feature representation and domain shift, not just ensemble selection.
- Evidence:
  - [Sandbox/output/prc_vs_prc2_feature_shift.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/prc_vs_prc2_feature_shift.csv) shows clear drift between `prc` and `prc2`.
  - The strongest shifts are in RGB intensity and ratio features that currently drive pool matching and regression quality.

## Next Steps

### 1. Stabilize the feature contract
- Keep a single canonical definition for pool features and global-model features in `Sandbox/common.py`.
- Avoid manual feature assembly in inference paths.
- Before new experiments, confirm train and infer use the same ordered column list.

### 2. Redesign `prc2` feature representation
- Add or revise features that are more robust to device/domain change:
  - normalized RGB intensity features
  - device-robust ratios and long/short-window contrasts
  - explicit quality features used by both pool and global branches
- Compare `prc2` performance with and without each new feature block instead of adding many changes at once.

### 3. Separate recovery experiments by layer
- Feature-layer experiments:
  - normalization strategy
  - revised pool feature set
  - revised global feature set
- Model-layer experiments:
  - `prc2`-only global model variants
  - stacking only after feature changes show a positive signal

### 4. Use a hard acceptance rule
- Accept a change only if it improves `prc2-*` toward `R2 > 0.1` without materially breaking the current `prc-*` results.
- Record each accepted or rejected experiment in `KnowledgeBase/findings.md` or `KnowledgeBase/pitfalls.md`.

## Immediate Implementation Target
- Refactor `Sandbox/common.py` so feature definitions for training, pool matching, and global blending are centrally declared and reused.
- Then run `prc2`-only inference first for faster iteration before rechecking full six-dataset summaries.
