"""
Run inference with the selected model pool and export per-dataset summaries.
"""
import json
import os

import joblib
import pandas as pd

from common import (
    MODEL_DIR,
    OUTPUT_DIR,
    ensemble_predict_and_evaluate,
    filter_feat_df_by_spo2_range,
    load_features_from_csv_paths,
)

DEFAULT_OUTPUT_DIR = MODEL_DIR
MODEL_POOL_FILENAME = "model_pool.joblib"
RAW_MODEL_POOL_FILENAME = "model_pool_raw.joblib"
CONFIG_FILENAME = "model_pool_config.json"

INFER_CSV_PATHS = [
    "./data/prc-c920.csv",
    "./data/prc-i15.csv",
    "./data/prc-i15m.csv",
    "./data/prc2-c930.csv",
    "./data/prc2-i16.csv",
    "./data/prc2-i16m.csv",
]

SAVE_PLOTS = True
USE_DOMAIN_SPECIALIZED_RAW_POOL = True
INFER_TOP_K_OVERRIDE = 20


def resolve_csv_paths(csv_paths):
    resolved_paths = []
    for csv_path in csv_paths:
        candidates = [csv_path]
        if csv_path.startswith("./data/"):
            candidates.append(csv_path.replace("./data/", "./data_new/", 1))
        resolved_path = next((path for path in candidates if os.path.isfile(path)), None)
        resolved_paths.append(resolved_path or csv_path)
    return resolved_paths


def load_model_and_config(output_dir=None):
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    model_path = os.path.join(output_dir, MODEL_POOL_FILENAME)
    raw_model_path = os.path.join(output_dir, RAW_MODEL_POOL_FILENAME)
    config_path = os.path.join(output_dir, CONFIG_FILENAME)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model pool not found: {model_path}. Run train.py first.")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}. Run train.py first.")

    model_pool = joblib.load(model_path)
    raw_model_pool = joblib.load(raw_model_path) if os.path.isfile(raw_model_path) else model_pool
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return model_pool, raw_model_pool, config


def dataset_prefix(name):
    return str(name).split("-", 1)[0] if name else ""


def specialize_model_pool(model_pool, infer_csv_paths):
    target_names = [os.path.splitext(os.path.basename(path))[0] for path in infer_csv_paths]
    target_prefixes = {dataset_prefix(name) for name in target_names}
    specialized_pool = [m for m in model_pool if dataset_prefix(m.get("dataset_name")) in target_prefixes]
    return specialized_pool, target_prefixes


def main():
    selected_model_pool, raw_model_pool, config = load_model_and_config(DEFAULT_OUTPUT_DIR)
    segment_length = config["SEGMENT_LENGTH"]
    top_k = INFER_TOP_K_OVERRIDE if INFER_TOP_K_OVERRIDE is not None else config["TOP_K"]
    use_normalization = config["USE_NORMALIZATION"]
    output_dir = config.get("OUTPUT_DIR", OUTPUT_DIR)
    data_csv_paths = config.get("DATA_CSV_PATHS", [])

    csv_paths = INFER_CSV_PATHS if INFER_CSV_PATHS is not None else data_csv_paths
    csv_paths = resolve_csv_paths(csv_paths)
    if not csv_paths:
        raise SystemExit("No inference CSV paths were provided.")

    active_model_pool = selected_model_pool
    pool_source = "selected"
    target_prefixes = set()
    if USE_DOMAIN_SPECIALIZED_RAW_POOL:
        specialized_pool, target_prefixes = specialize_model_pool(raw_model_pool, csv_paths)
        if len(specialized_pool) > 0:
            active_model_pool = specialized_pool
            pool_source = "raw_domain_specialized"

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    print("=" * 60)
    print(f"Loaded {pool_source} model pool with {len(active_model_pool)} models")
    if target_prefixes:
        print(f"Target dataset prefixes: {sorted(target_prefixes)}")
    print("=" * 60)

    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Skip missing CSV: {csv_path}")
            continue

        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        print("\n" + "=" * 60)
        print(f"Running inference for {csv_path}")
        print("=" * 60)

        feat_df = load_features_from_csv_paths([csv_path], verbose=True)
        feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=segment_length, verbose=True)
        if len(feat_df) == 0:
            summary_rows.append({
                "csv_path": csv_path,
                "csv_name": csv_name,
                "pool_source": pool_source,
                "pool_size": len(active_model_pool),
                "n_subjects": 0,
                "mean_PCC": float("nan"),
                "overall_PCC": float("nan"),
                "mean_R2": float("nan"),
                "mean_MAE": float("nan"),
            })
            continue

        results_df = ensemble_predict_and_evaluate(
            model_pool=active_model_pool,
            feat_df=feat_df,
            selected_channels=None,
            top_k=top_k,
            segment_length=segment_length,
            use_normalization=use_normalization,
            save_plots=SAVE_PLOTS,
            output_dir=output_dir,
        )
        if len(results_df) == 0:
            summary_rows.append({
                "csv_path": csv_path,
                "csv_name": csv_name,
                "pool_source": pool_source,
                "pool_size": len(active_model_pool),
                "n_subjects": 0,
                "mean_PCC": float("nan"),
                "overall_PCC": float("nan"),
                "mean_R2": float("nan"),
                "mean_MAE": float("nan"),
            })
            continue

        per_subject_cols = ["subject_id", "n_frames", "R2", "MAE", "PCC"]
        per_subject_df = results_df[[c for c in per_subject_cols if c in results_df.columns]].copy()
        per_subject_path = os.path.join(
            output_dir, f"results_per_subject_{csv_name}_topk{top_k}_seg{segment_length}.csv"
        )
        per_subject_df.to_csv(per_subject_path, index=False, encoding="utf-8")

        mean_pcc = results_df["PCC"].mean()
        overall_pcc = results_df["overall_PCC"].iloc[0] if "overall_PCC" in results_df.columns else None
        mean_r2 = results_df["R2"].mean()
        mean_mae = results_df["MAE"].mean()

        summary_rows.append({
            "csv_path": csv_path,
            "csv_name": csv_name,
            "pool_source": pool_source,
            "pool_size": len(active_model_pool),
            "n_subjects": len(results_df),
            "mean_PCC": mean_pcc,
            "overall_PCC": overall_pcc,
            "mean_R2": mean_r2,
            "mean_MAE": mean_mae,
        })

        overall_pcc_str = f"{overall_pcc:.4f}" if overall_pcc is not None and not pd.isna(overall_pcc) else "N/A"
        print(f"[{csv_name}] mean PCC={mean_pcc:.4f}, overall PCC={overall_pcc_str}, mean R2={mean_r2:.4f}, mean MAE={mean_mae:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"infer_summary_topk{top_k}_seg{segment_length}.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("Inference summary")
    print("=" * 60)
    if len(summary_df) > 0:
        print(summary_df.to_string(index=False))
    print(f"Saved summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
