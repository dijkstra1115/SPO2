# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Remote photoplethysmography (rPPG) ensemble learning system for non-contact blood oxygen (SpO2) estimation from RGB video. Trains a pool of segment-level linear models from RGB camera data, then selects the most similar models at inference time via multi-dimensional similarity matching.

## Commands

```bash
# Always use conda env RGB-SpO2 to run python
conda activate RGB-SpO2

# Training — builds model pool from configured datasets
python train.py

# Inference — runs ensemble prediction on test datasets
python infer.py

# Install dependencies
pip install -r requirements.txt
```

## Development Guidelines

- Use `conda activate RGB-SpO2` before running any Python commands
- Use "R2" instead of "R^2" or "R²" in output/code to avoid Windows UTF encoding issues
- Run shell commands in the background when they take a long time (training, inference)


## Architecture

**Three-phase ML pipeline:**

1. **Feature Engineering** (`build_features_from_df` in `common.py`): Extracts rPPG signal via POS algorithm from RGB frames → FFT-based heart rate detection → dynamic bandpass filtering → rolling-window AC/DC ratio features per channel. Operates per-folder to prevent filter state bleeding.

2. **Model Pool Training** (`train_model_pool` in `common.py`): Segments data into 900-frame chunks. Trains independent Linear Regression models per segment. Quality-filters by R² > 0.5 AND PCC > 0.5. Stores models with their feature segments and RGB means for similarity lookup.

3. **Ensemble Inference** (`ensemble_predict_and_evaluate` in `common.py`): For each test segment, computes shape score (Pearson correlation), range score (CCC), and RGB score (cosine similarity of lighting). Two-stage ranking selects TOP_K=5 models. Ensemble-averages their predictions.

## Key Configuration

All tunable constants live at the top of `common.py` (lines 19-58):
- `WIN_LEN=60`, `FS=30`, `SEGMENT_LENGTH=900` — signal processing and segmentation
- `TOP_K=5` — ensemble size
- `MIN_TRAIN_R2=0.5`, `MIN_TRAIN_PCC=0.5` — model quality thresholds
- `MIN_SpO2_RANGE=10` — subject filtering (must have ≥10% SpO2 spread)

## Naming Convention

- SpO2 label column: `SpO2_win_last` — the last frame's SpO2 value within each rolling window (not a mean)
- `RGB_SIM_WEIGHT=0.5` — weight of lighting similarity in model selection

Dataset paths are configured in `train.py` (`DATA_CSV_PATHS`) and `infer.py` (`INFER_CSV_PATHS`).

## Data Format

Input CSVs have columns: `Folder` (subject ID), `COLOR_R`, `COLOR_G`, `COLOR_B` (0-255), `SpO2` (reference %). Sampled at 30 fps. Datasets span multiple devices: C920, C930, iPhone 15 Pro/Max, iPhone 16 Pro/Max.

## Output

- Model pool: `output/model/model_pool.joblib` + `model_pool_config.json`
- Per-subject plots: `output/sub_plot/vis_ensemble_{subject}_topk{k}.png`
- CSV results: `output/` directory

## Knowledge Base

`knowledge/` folder contains experiment findings, pitfalls, and best configurations. **Read before starting new experiments** to avoid repeating failed approaches.

- **Entry point**: `knowledge/index.md` — links to all topic files
- **Before running a new experiment**: Check `knowledge/experiment_index.md` for what's been tried and `knowledge/calibration_pitfalls.md` / `knowledge/c930_analysis.md` for known dead ends
- **After completing an experiment**: Update `knowledge/best_results.md` if a new best is found, and add a line to `knowledge/experiment_index.md`
- **Key takeaways to remember**:
  - seg=600 is optimal (see `knowledge/segment_length.md`)
  - Device normalization is essential (see `knowledge/device_normalization.md`)
  - Post-hoc calibration fails for low-sample devices (see `knowledge/calibration_pitfalls.md`)
  - c930 R2 is capped by 2 outlier subjects (see `knowledge/c930_analysis.md`)
