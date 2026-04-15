import json
import os

import joblib
import pandas as pd

import common
from common import ensemble_predict_and_evaluate, filter_feat_df_by_spo2_range, load_features_from_csv_paths

MODEL_DIR = os.path.join(common.OUTPUT_DIR, "model")
RAW_POOL_PATH = os.path.join(MODEL_DIR, "model_pool_raw.joblib")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_pool_config.json")

CSV_PATHS = [
    "./data_new/prc-c920.csv",
    "./data_new/prc-i15.csv",
    "./data_new/prc-i15m.csv",
]

EXPERIMENTS = [
    {"name": "raw_all_topk20", "pool_filter": None, "top_k": 20, "global_blend": 0.35},
    {"name": "raw_all_topk40", "pool_filter": None, "top_k": 40, "global_blend": 0.35},
    {"name": "raw_prc_topk20", "pool_filter": {"prc-c920", "prc-i15", "prc-i15m"}, "top_k": 20, "global_blend": 0.35},
    {"name": "raw_prc_topk40", "pool_filter": {"prc-c920", "prc-i15", "prc-i15m"}, "top_k": 40, "global_blend": 0.35},
]


def run_experiment(raw_pool, config, experiment):
    common.GLOBAL_MODEL_BLEND = experiment["global_blend"]
    segment_length = config["SEGMENT_LENGTH"]
    use_normalization = config["USE_NORMALIZATION"]

    if experiment["pool_filter"] is None:
        model_pool = raw_pool
    else:
        allowed = experiment["pool_filter"]
        model_pool = [m for m in raw_pool if m.get("dataset_name") in allowed]

    summary_rows = []
    print("=" * 60)
    print(f"Experiment: {experiment['name']}, pool_size={len(model_pool)}, top_k={experiment['top_k']}, global_blend={experiment['global_blend']}")
    print("=" * 60)

    for csv_path in CSV_PATHS:
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        feat_df = load_features_from_csv_paths([csv_path], verbose=False)
        feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=segment_length, verbose=False)
        results_df = ensemble_predict_and_evaluate(
            model_pool=model_pool,
            feat_df=feat_df,
            selected_channels=None,
            top_k=experiment["top_k"],
            segment_length=segment_length,
            use_normalization=use_normalization,
            save_plots=False,
            output_dir=common.OUTPUT_DIR,
        )
        mean_pcc = results_df["PCC"].mean() if len(results_df) else float("nan")
        overall_pcc = results_df["overall_PCC"].iloc[0] if len(results_df) else float("nan")
        mean_r2 = results_df["R2"].mean() if len(results_df) else float("nan")
        mean_mae = results_df["MAE"].mean() if len(results_df) else float("nan")
        summary_rows.append({
            "experiment": experiment["name"],
            "csv_name": csv_name,
            "pool_size": len(model_pool),
            "top_k": experiment["top_k"],
            "global_blend": experiment["global_blend"],
            "n_subjects": len(results_df),
            "mean_PCC": mean_pcc,
            "overall_PCC": overall_pcc,
            "mean_R2": mean_r2,
            "mean_MAE": mean_mae,
        })
        print(f"{csv_name}: mean_R2={mean_r2:.4f}, mean_PCC={mean_pcc:.4f}, overall_PCC={overall_pcc:.4f}, mean_MAE={mean_mae:.4f}")

    return pd.DataFrame(summary_rows)


def main():
    raw_pool = joblib.load(RAW_POOL_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    result_frames = []
    for experiment in EXPERIMENTS:
        result_frames.append(run_experiment(raw_pool, config, experiment))

    summary_df = pd.concat(result_frames, ignore_index=True)
    output_path = os.path.join(common.OUTPUT_DIR, "pool_experiment_summary.csv")
    summary_df.to_csv(output_path, index=False, encoding="utf-8")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"Saved experiment summary: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
