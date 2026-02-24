"""
推論腳本：載入已儲存的 model pool 與設定，對「每個 CSV 分別」做 ensemble 推論，
並將各 CSV 的 平均 PCC、整體 PCC、平均 R²、平均 MAE 彙總成一個 .csv。
"""
import os
import json
import joblib
import pandas as pd

from common import (
    OUTPUT_DIR,
    MODEL_DIR,
    load_features_from_csv_paths,
    ensemble_predict_and_evaluate,
)

# 預設從訓練時儲存的目錄載入（output/model/）
DEFAULT_OUTPUT_DIR = MODEL_DIR
MODEL_POOL_FILENAME = "model_pool.joblib"
CONFIG_FILENAME = "model_pool_config.json"

# 推論用的 CSV 路徑列表；每個 CSV 單獨推論一次，不合併
# 若為 None，則使用訓練時儲存在 config 裡的 DATA_CSV_PATHS
# INFER_CSV_PATHS = None
# 例如:
INFER_CSV_PATHS = ["./data/prc-c920.csv", "./data/prc-i15.csv", "./data/prc-i15m.csv"]

# 是否儲存每個 subject 的預測圖
SAVE_PLOTS = True


def load_model_and_config(output_dir=None):
    """載入模型池與設定。"""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    model_path = os.path.join(output_dir, MODEL_POOL_FILENAME)
    config_path = os.path.join(output_dir, CONFIG_FILENAME)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型檔: {model_path}，請先執行 train.py")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到設定檔: {config_path}，請先執行 train.py")
    model_pool = joblib.load(model_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return model_pool, config


def main():
    model_pool, config = load_model_and_config(DEFAULT_OUTPUT_DIR)
    segment_length = config["SEGMENT_LENGTH"]
    top_k = config["TOP_K"]
    use_normalization = config["USE_NORMALIZATION"]
    output_dir = config.get("OUTPUT_DIR", OUTPUT_DIR)
    data_csv_paths = config.get("DATA_CSV_PATHS", [])

    csv_paths = INFER_CSV_PATHS if INFER_CSV_PATHS is not None else data_csv_paths
    if not csv_paths:
        raise SystemExit("沒有指定推論用 CSV（INFER_CSV_PATHS 與 config 的 DATA_CSV_PATHS 皆為空）。")

    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []

    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"跳過（檔案不存在）: {csv_path}")
            continue

        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        print("\n" + "=" * 60)
        print(f"推論 CSV: {csv_path}")
        print("=" * 60)

        feat_df = load_features_from_csv_paths([csv_path], verbose=True)
        if len(feat_df) == 0:
            print(f"  沒有樣本，跳過。")
            summary_rows.append({
                "csv_path": csv_path,
                "csv_name": csv_name,
                "n_subjects": 0,
                "mean_PCC": float("nan"),
                "overall_PCC": float("nan"),
                "mean_R2": float("nan"),
                "mean_MAE": float("nan"),
            })
            continue

        results_df = ensemble_predict_and_evaluate(
            model_pool=model_pool,
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
                "n_subjects": 0,
                "mean_PCC": float("nan"),
                "overall_PCC": float("nan"),
                "mean_R2": float("nan"),
                "mean_MAE": float("nan"),
            })
            continue

        # 輸出每個 subject 的結果：subject_id, n_segments, R2, MAE, PCC
        per_subject_cols = ["subject_id", "n_segments", "R2", "MAE", "PCC"]
        per_subject_df = results_df[[c for c in per_subject_cols if c in results_df.columns]].copy()
        per_subject_path = os.path.join(
            output_dir, f"results_per_subject_{csv_name}_topk{top_k}_seg{segment_length}.csv"
        )
        per_subject_df.to_csv(per_subject_path, index=False)
        print(f"  每個 subject 結果已儲存: {per_subject_path}")

        mean_pcc = results_df["PCC"].mean()
        overall_pcc = results_df["overall_PCC"].iloc[0] if "overall_PCC" in results_df.columns else None
        mean_r2 = results_df["R2"].mean()
        mean_mae = results_df["MAE"].mean()
        n_subjects = len(results_df)

        summary_rows.append({
            "csv_path": csv_path,
            "csv_name": csv_name,
            "n_subjects": n_subjects,
            "mean_PCC": mean_pcc,
            "overall_PCC": overall_pcc,
            "mean_R2": mean_r2,
            "mean_MAE": mean_mae,
        })

        overall_pcc_str = f"{overall_pcc:.4f}" if overall_pcc is not None and not pd.isna(overall_pcc) else "N/A"
        print(f"\n  [{csv_name}] 平均 PCC = {mean_pcc:.4f}, 整體 PCC = {overall_pcc_str}, 平均 R² = {mean_r2:.4f}, 平均 MAE = {mean_mae:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"infer_summary_topk{top_k}_seg{segment_length}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("各 CSV 推論結果彙總")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\n彙總已儲存: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
