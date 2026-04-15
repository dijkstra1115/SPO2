"""
Train the model pool from one or more CSV datasets and persist diagnostics.
"""
import json
import os

import joblib
import pandas as pd

from common import (
    ALPHA,
    BW,
    FS,
    HR_UPDATE_SEC,
    MIN_RPPG_FRAMES,
    MODEL_DIR,
    OUTPUT_DIR,
    SEGMENT_LENGTH,
    STEP,
    TOP_K,
    USE_NORMALIZATION,
    USE_REGULARIZATION,
    WIN_LEN,
    all_channel_names,
    filter_feat_df_by_spo2_range,
    load_features_from_csv_paths,
    select_model_pool,
    train_model_pool,
)

DATA_CSV_PATHS = [
    "./data/prc-c920.csv",
    "./data/prc-i15.csv",
    "./data/prc-i15m.csv",
    "./data/prc2-c930.csv",
    "./data/prc2-i16.csv",
    "./data/prc2-i16m.csv",
]

MODEL_POOL_FILENAME = "model_pool.joblib"
RAW_MODEL_POOL_FILENAME = "model_pool_raw.joblib"
CONFIG_FILENAME = "model_pool_config.json"
MODEL_DATASET_EVAL_FILENAME = "model_pool_dataset_eval.csv"
MODEL_SUMMARY_FILENAME = "model_pool_summary.csv"

MODEL_SELECTION_MIN_DATASET_R2 = 0.1
MODEL_SELECTION_MIN_POOL_SIZE = 6
MODEL_SELECTION_MAX_POOL_SIZE = 18


def resolve_csv_paths(csv_paths):
    resolved_paths = []
    for csv_path in csv_paths:
        candidates = [csv_path]
        if csv_path.startswith("./data/"):
            candidates.append(csv_path.replace("./data/", "./data_new/", 1))
        resolved_path = next((path for path in candidates if os.path.isfile(path)), None)
        resolved_paths.append(resolved_path or csv_path)
    return resolved_paths


def build_config(data_csv_paths, raw_pool_size, selected_pool_size):
    return {
        "SEGMENT_LENGTH": SEGMENT_LENGTH,
        "TOP_K": TOP_K,
        "USE_NORMALIZATION": USE_NORMALIZATION,
        "USE_REGULARIZATION": USE_REGULARIZATION,
        "ALPHA": ALPHA,
        "all_channel_names": list(all_channel_names),
        "WIN_LEN": WIN_LEN,
        "STEP": STEP,
        "FS": FS,
        "HR_UPDATE_SEC": HR_UPDATE_SEC,
        "MIN_RPPG_FRAMES": MIN_RPPG_FRAMES,
        "BW": BW,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MODEL_DIR": MODEL_DIR,
        "DATA_CSV_PATHS": data_csv_paths,
        "MODEL_SELECTION_MIN_DATASET_R2": MODEL_SELECTION_MIN_DATASET_R2,
        "MODEL_SELECTION_MIN_POOL_SIZE": MODEL_SELECTION_MIN_POOL_SIZE,
        "MODEL_SELECTION_MAX_POOL_SIZE": MODEL_SELECTION_MAX_POOL_SIZE,
        "raw_model_pool_size": raw_pool_size,
        "selected_model_pool_size": selected_pool_size,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_csv_paths = resolve_csv_paths(DATA_CSV_PATHS)
    print("=" * 60)
    print("Loading CSV feature sources...")
    print("=" * 60)
    feat_df_all = load_features_from_csv_paths(data_csv_paths, verbose=True)
    if len(feat_df_all) == 0:
        raise SystemExit("No features were built. Check DATA_CSV_PATHS and dataset availability.")

    print(f"Total rows: {len(feat_df_all)}, subjects: {feat_df_all['subject_id'].nunique()}")
    print("=" * 60)
    print("Filtering subjects by minimum SpO2 range...")
    feat_df_all = filter_feat_df_by_spo2_range(feat_df_all, segment_length=SEGMENT_LENGTH, verbose=True)
    if len(feat_df_all) == 0:
        raise SystemExit("No subjects left after SpO2 range filtering.")

    print(f"Filtered rows: {len(feat_df_all)}, subjects: {feat_df_all['subject_id'].nunique()}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Building raw model pool")
    print("=" * 60)
    raw_model_pool = train_model_pool(
        selected_channels=None,
        feat_df=feat_df_all,
        segment_length=SEGMENT_LENGTH,
        use_normalization=USE_NORMALIZATION,
        use_regularization=USE_REGULARIZATION,
        alpha=ALPHA,
    )

    print("\n" + "=" * 60)
    print("Selecting dataset-aware model pool")
    print("=" * 60)
    model_pool, dataset_eval_df, model_summary_df = select_model_pool(
        raw_model_pool=raw_model_pool,
        feat_df=feat_df_all,
        selected_channels=None,
        segment_length=SEGMENT_LENGTH,
        use_normalization=USE_NORMALIZATION,
        min_dataset_r2=MODEL_SELECTION_MIN_DATASET_R2,
        min_pool_size=MODEL_SELECTION_MIN_POOL_SIZE,
        max_pool_size=MODEL_SELECTION_MAX_POOL_SIZE,
    )

    if len(dataset_eval_df) == 0:
        dataset_eval_df = pd.DataFrame()
    if len(model_summary_df) == 0:
        model_summary_df = pd.DataFrame()

    config = build_config(data_csv_paths, len(raw_model_pool), len(model_pool))
    model_path = os.path.join(MODEL_DIR, MODEL_POOL_FILENAME)
    raw_model_path = os.path.join(MODEL_DIR, RAW_MODEL_POOL_FILENAME)
    config_path = os.path.join(MODEL_DIR, CONFIG_FILENAME)
    dataset_eval_path = os.path.join(MODEL_DIR, MODEL_DATASET_EVAL_FILENAME)
    model_summary_path = os.path.join(MODEL_DIR, MODEL_SUMMARY_FILENAME)

    joblib.dump(model_pool, model_path)
    joblib.dump(raw_model_pool, raw_model_path)
    dataset_eval_df.to_csv(dataset_eval_path, index=False, encoding="utf-8")
    model_summary_df.to_csv(model_summary_path, index=False, encoding="utf-8")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Training finished")
    print("=" * 60)
    print(f"Raw model pool:      {raw_model_path} ({len(raw_model_pool)} models)")
    print(f"Selected model pool: {model_path} ({len(model_pool)} models)")
    print(f"Dataset diagnostics: {dataset_eval_path}")
    print(f"Model summary:       {model_summary_path}")
    print(f"Config:              {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
