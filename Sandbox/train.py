"""
訓練腳本：載入多個 CSV、建特徵、訓練 model pool，並將模型池與設定儲存供推論使用。
"""
import os
import json
import joblib

from common import (
    OUTPUT_DIR,
    MODEL_DIR,
    SEGMENT_LENGTH,
    TOP_K,
    USE_NORMALIZATION,
    USE_REGULARIZATION,
    ALPHA,
    WIN_LEN,
    STEP,
    FS,
    HR_UPDATE_SEC,
    MIN_RPPG_FRAMES,
    BW,
    all_channel_names,
    load_features_from_csv_paths,
    filter_feat_df_by_spo2_range,
    train_model_pool,
)

# 多 CSV 擴大模型池：可一次加入多個 .csv，合併後訓練
DATA_CSV_PATHS = [
    "./data_new/prc2-c930.csv",
    "./data_new/prc2-i16.csv",
    "./data_new/prc2-i16m.csv",
]

# 訓練時隨機反轉的 folder 比例
REVERSE_RATIO = 0.0

# 儲存檔名（可改為參數）
MODEL_POOL_FILENAME = "model_pool.joblib"
CONFIG_FILENAME = "model_pool_config.json"


def get_config():
    """產生與訓練時一致的設定，供儲存與推論載入使用。"""
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
        "DATA_CSV_PATHS": DATA_CSV_PATHS,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("載入 CSV 並建特徵...")
    print("=" * 60)
    feat_df_all = load_features_from_csv_paths(DATA_CSV_PATHS, verbose=True)
    if len(feat_df_all) == 0:
        raise SystemExit("沒有載入到任何樣本，請檢查 DATA_CSV_PATHS 與檔案是否存在。")
    print(f"總樣本數: {len(feat_df_all)}, 總 subject 數: {feat_df_all['subject_id'].nunique()}")
    print("=" * 60)
    print("依 SpO2 跨幅篩選 subject（僅保留跨幅 >= 10，且僅以長度足夠的 folder 計算跨幅）...")
    feat_df_all = filter_feat_df_by_spo2_range(feat_df_all, segment_length=SEGMENT_LENGTH, verbose=True)
    if len(feat_df_all) == 0:
        raise SystemExit("篩選後沒有剩餘樣本，請檢查 MIN_SpO2_RANGE 或資料。")
    print(f"篩選後樣本數: {len(feat_df_all)}, subject 數: {feat_df_all['subject_id'].nunique()}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Ensemble Model Pool 訓練")
    print("=" * 60)
    model_pool = train_model_pool(
        selected_channels=None,
        feat_df=feat_df_all,
        segment_length=SEGMENT_LENGTH,
        use_normalization=USE_NORMALIZATION,
        use_regularization=USE_REGULARIZATION,
        alpha=ALPHA,
        reverse_ratio=REVERSE_RATIO,
    )

    config = get_config()
    model_path = os.path.join(MODEL_DIR, MODEL_POOL_FILENAME)
    config_path = os.path.join(MODEL_DIR, CONFIG_FILENAME)

    joblib.dump(model_pool, model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("訓練完成，已儲存模型與設定")
    print("=" * 60)
    print(f"  模型池: {model_path} ({len(model_pool)} 個模型)")
    print(f"  設定:   {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
