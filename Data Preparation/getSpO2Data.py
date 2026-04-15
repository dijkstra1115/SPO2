import pandas as pd
import numpy as np
import os
from typing import List, Optional

SpO2_MIN = 50
SpO2_MAX = 100
IQA_MOTION_THRESHOLD = 0.9

# 所有資料集的設定
DATASETS = [
    {"name": "prc-c920",  "data_dir": "//172.16.1.2/Algorithm/PRC SpO2 Data",   "dump_dir": "Dump",        "device": "C920",                    "meta_file": "Meta.csv"},
    {"name": "prc-i15",   "data_dir": "//172.16.1.2/Algorithm/PRC SpO2 Data",   "dump_dir": "Dump",        "device": "iPhone 15 Pro-Front",     "meta_file": "Meta.csv"},
    {"name": "prc-i15m",  "data_dir": "//172.16.1.2/Algorithm/PRC SpO2 Data",   "dump_dir": "Dump",        "device": "iPhone 15 Pro Max-Front", "meta_file": "Meta.csv"},
    {"name": "prc2-c930", "data_dir": "//172.16.1.2/Algorithm/PRC SpO2 (2025)", "dump_dir": "Dump(5.8.4)", "device": "C930",                    "meta_file": "meta.csv"},
    {"name": "prc2-i16",  "data_dir": "//172.16.1.2/Algorithm/PRC SpO2 (2025)", "dump_dir": "Dump(5.8.4)", "device": "iPhone 16 Pro-Front",     "meta_file": "meta.csv"},
    {"name": "prc2-i16m", "data_dir": "//172.16.1.2/Algorithm/PRC SpO2 (2025)", "dump_dir": "Dump(5.8.4)", "device": "iPhone 16 Pro Max-Front", "meta_file": "meta.csv"},
]

def process_folder_data(folder_name: str, data_dir: str, dump_dir: str = "Dump", device: str = "C920") -> Optional[List[pd.DataFrame]]:
    """
    處理單一 Folder 的資料，合併 {device}.csv 和 masimo.csv

    過濾條件：
    1. SpO2 資料線性內插成與 dump 相同的單位（偵）
    2. 將連續的 IQA Motion >= 0.9 的數據視為獨立序列
    3. 跳過 IQA Motion < 0.9 的數據段
    4. 只保留長度 >= 30 偵的序列
    5. 只保留 SpO2 在 70-99 範圍內的資料

    Args:
        folder_name: Folder 名稱
        data_dir: 資料根目錄
        dump_dir: Dump 子目錄名稱
        device: 裝置名稱

    Returns:
        包含多個序列的 DataFrame 列表，或 None (如果處理失敗)
    """
    try:
        # 讀取 {device}.csv (RGB 資料 + IQA Motion)
        dump_path = os.path.join(data_dir, dump_dir, folder_name, f"{device}.csv")
        dump_data = pd.read_csv(dump_path, usecols=["COLOR_R", "COLOR_G", "COLOR_B", "IQA Motion"])
        print(f"  - {device}.csv: {len(dump_data)} 偵")

        # 讀取 masimo.csv (SpO2 資料)
        masimo_path = os.path.join(data_dir, "Label", folder_name, "masimo.csv")
        masimo_data = pd.read_csv(masimo_path, usecols=["SPO2"])
        print(f"  - masimo.csv: {len(masimo_data)} 秒")

        # 處理 SpO2 缺失值 (將 "--" 等非數值轉換為 NaN)
        masimo_data["SPO2"] = pd.to_numeric(masimo_data["SPO2"], errors='coerce')
        missing_count = masimo_data["SPO2"].isna().sum()
        if missing_count > 0:
            print(f"  - SpO2 缺失值: {missing_count}/{len(masimo_data)} 秒")

        # 檢查 SpO2 資料是否有效
        if masimo_data["SPO2"].isna().all():
            print(f"  - 警告: {folder_name} 的 SpO2 資料全部缺失")
            return None

        # 檢查是否有 SpO2 在指定範圍內的數據
        spo2_in_range = (masimo_data["SPO2"].dropna() >= SpO2_MIN) & (masimo_data["SPO2"].dropna() <= SpO2_MAX)
        if not spo2_in_range.any():
            print(f"  - 跳過 {folder_name}: 沒有 SpO2 在 {SpO2_MIN}-{SpO2_MAX} 範圍內的數據")
            return None

        # 排除最後一秒的資料 (避免不完整的 30偵)
        masimo_data = masimo_data.iloc[:-1]
        print(f"  - 排除最後一秒後: {len(masimo_data)} 秒")

        # 計算對應的 dump 資料長度 (每秒30偵)
        expected_frames = len(masimo_data) * 30

        # 確保 dump 資料長度足夠
        if len(dump_data) < expected_frames:
            print(f"  - 警告: dump 資料不足 (需要 {expected_frames} 偵，實際 {len(dump_data)} 偵)")
            # 調整 masimo 資料長度以匹配 dump
            masimo_data = masimo_data.iloc[:len(dump_data)//30]
            expected_frames = len(masimo_data) * 30

        # 截取對應長度的 dump 資料
        dump_data = dump_data.iloc[:expected_frames]

        # 將 SpO2 資料線性內插到每偵（30Hz）
        spo2_1hz = masimo_data["SPO2"].values.astype(float)
        n_seconds = len(spo2_1hz)
        # masimo[i] 對應第 i 秒的起始幀 (frame i*30)
        masimo_frame_idx = np.arange(n_seconds) * 30
        frame_idx = np.arange(expected_frames)

        valid_1hz = ~np.isnan(spo2_1hz)
        if valid_1hz.sum() >= 2:
            # 在有效值之間線性內插
            spo2_expanded = np.interp(frame_idx, masimo_frame_idx[valid_1hz], spo2_1hz[valid_1hz])
            # 將原本 NaN 秒對應的 30 幀標記回 NaN（不跨 NaN 插值）
            for i in range(n_seconds):
                if np.isnan(spo2_1hz[i]):
                    start = i * 30
                    end = min((i + 1) * 30, expected_frames)
                    spo2_expanded[start:end] = np.nan
        else:
            spo2_expanded = np.full(expected_frames, np.nan)

        # 找出 SpO2 缺失的位置
        spo2_valid_mask = ~np.isnan(spo2_expanded)

        # 找出 IQA Motion >= 閾值的位置
        iqa_motion = dump_data["IQA Motion"].values
        iqa_valid_mask = iqa_motion >= IQA_MOTION_THRESHOLD

        # 找出 SpO2 在指定範圍內的位置
        spo2_range_mask = (spo2_expanded >= SpO2_MIN) & (spo2_expanded <= SpO2_MAX)

        # 結合所有過濾條件：SpO2 非缺失、IQA Motion 有效、SpO2 在指定範圍內
        valid_mask = spo2_valid_mask & iqa_valid_mask & spo2_range_mask

        # 找出連續的有效序列
        sequences = []
        current_start = None
        current_length = 0

        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                if current_start is None:
                    # 開始新的序列
                    current_start = i
                    current_length = 1
                else:
                    # 繼續當前序列
                    current_length += 1
            else:
                if current_start is not None:
                    # 結束當前序列
                    if current_length >= 30:  # 只保留長度 >= 30 的序列
                        sequences.append((current_start, current_start + current_length))
                        print(f"  - 找到有效序列: 第 {current_start+1}-{current_start + current_length} 偵 ({current_length} 偵)")
                    else:
                        print(f"  - 跳過短序列: 第 {current_start+1}-{current_start + current_length} 偵 ({current_length} 偵 < 30)")
                    current_start = None
                    current_length = 0

        # 處理最後一個序列
        if current_start is not None and current_length >= 30:
            sequences.append((current_start, current_start + current_length))
            print(f"  - 找到有效序列: 第 {current_start+1}-{current_start + current_length} 偵 ({current_length} 偵)")
        elif current_start is not None:
            print(f"  - 跳過短序列: 第 {current_start+1}-{current_start + current_length} 偵 ({current_length} 偵 < 30)")

        if not sequences:
            print(f"  - 警告: {folder_name} 沒有找到長度 >= 30 的有效序列")
            return None

        # 為每個序列建立 DataFrame
        result_dataframes = []
        for seq_idx, (start_idx, end_idx) in enumerate(sequences, 1):
            # 提取序列數據
            seq_dump = dump_data.iloc[start_idx:end_idx]
            seq_spo2 = spo2_expanded[start_idx:end_idx]

            # 建立序列的 DataFrame
            seq_df = pd.DataFrame({
                'Folder': [f"{folder_name}_seq{seq_idx}"] * len(seq_dump),
                'COLOR_R': seq_dump["COLOR_R"].values,
                'COLOR_G': seq_dump["COLOR_G"].values,
                'COLOR_B': seq_dump["COLOR_B"].values,
                'SpO2': seq_spo2
            })

            result_dataframes.append(seq_df)
            print(f"  - 序列 {seq_idx}: {len(seq_df)} 偵")

        return result_dataframes

    except FileNotFoundError as e:
        print(f"  - 錯誤: 找不到檔案 {e}")
        return None
    except Exception as e:
        print(f"  - 錯誤: 處理 {folder_name} 時發生異常: {e}")
        return None

def merge_all_folders(data_dir: str, output_path: str = "data.csv",
                      dump_dir: str = "Dump", device: str = "C920",
                      meta_file: str = "Meta.csv",
                      max_folders: Optional[int] = None) -> None:
    """
    處理所有 Folder 並合併成一個大的資料集

    Args:
        data_dir: 資料根目錄
        output_path: 輸出檔案路徑
        dump_dir: Dump 子目錄名稱
        device: 裝置名稱
        meta_file: Meta 檔案名稱
        max_folders: 最大處理的 Folder 數量，None 表示處理全部
    """
    # 讀取 Meta.csv
    meta_path = os.path.join(data_dir, meta_file)

    try:
        meta_df = pd.read_csv(meta_path)
        print(f"讀取 {meta_file}: {len(meta_df)} 個 Folder")
    except FileNotFoundError:
        print(f"錯誤: 找不到 {meta_path}")
        return

    if "Folder" not in meta_df.columns:
        print(f"錯誤: {meta_file} 中沒有 'Folder' 欄位")
        return

    # 獲取所有 Folder 清單
    folders = meta_df["Folder"].unique()
    total_folders = len(folders)

    # 如果指定了最大處理數量，則限制 Folder 數量
    if max_folders is not None and max_folders < total_folders:
        folders = folders[:max_folders]
        print(f"總 Folder 數量: {total_folders} 個")
        print(f"限制處理數量: {max_folders} 個")
        print(f"實際處理的 Folder: {len(folders)} 個")
    else:
        print(f"需要處理的 Folder: {total_folders} 個")

    # 處理每個 Folder
    all_data = []
    successful_folders = 0

    for i, folder in enumerate(folders, 1):
        print(f"\n處理 Folder {i}/{len(folders)}: {folder}")

        folder_sequences = process_folder_data(folder, data_dir, dump_dir=dump_dir, device=device)

        if folder_sequences is not None:
            all_data.extend(folder_sequences)
            successful_folders += 1
            print(f"  - 成功處理，找到 {len(folder_sequences)} 個序列，總資料筆數: {sum(len(seq) for seq in folder_sequences)}")
        else:
            print(f"  - 處理失敗，跳過")

    # 合併所有資料
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"\n合併完成:")
        print(f"  - 成功處理的 Folder: {successful_folders}/{len(folders)}")
        if max_folders is not None and max_folders < total_folders:
            print(f"  - 總 Folder 數量: {total_folders} 個 (已限制處理 {max_folders} 個)")
        print(f"  - 總資料筆數: {len(final_df)}")
        print(f"  - 欄位: {list(final_df.columns)}")

        # 顯示每個序列的資料統計
        sequence_stats = final_df.groupby('Folder').size()
        print(f"  - 每個序列的資料筆數:")
        for sequence, count in sequence_stats.items():
            print(f"    {sequence}: {count} 筆")

        # 統計原始 Folder 的序列數量
        original_folders = {}
        for sequence in sequence_stats.index:
            original_folder = sequence.split('_seq')[0]
            if original_folder not in original_folders:
                original_folders[original_folder] = 0
            original_folders[original_folder] += 1

        print(f"  - 每個原始 Folder 的序列數量:")
        for folder, seq_count in original_folders.items():
            print(f"    {folder}: {seq_count} 個序列")

        # 輸出到檔案
        final_df.to_csv(output_path, index=False)
        print(f"\n資料已保存到: {output_path}")

    else:
        print("\n錯誤: 沒有成功處理任何 Folder")

if __name__ == "__main__":
    import sys

    OUTPUT_DIR = "../Sandbox/data_new"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 設定最大處理的 Folder 數量 (None 表示處理全部)
    MAX_FOLDERS = None

    # 可指定要跑的 dataset name，例如: python getSpO2Data.py prc-c920 prc2-c930
    # 不指定則全部跑
    if len(sys.argv) > 1:
        names_to_run = sys.argv[1:]
    else:
        names_to_run = [d["name"] for d in DATASETS]

    for ds in DATASETS:
        if ds["name"] not in names_to_run:
            continue
        output_path = os.path.join(OUTPUT_DIR, f"{ds['name']}.csv")
        print("\n" + "=" * 70)
        print(f"處理資料集: {ds['name']} → {output_path}")
        print(f"  DATA_DIR: {ds['data_dir']}")
        print(f"  DEVICE:   {ds['device']}")
        print("=" * 70)
        merge_all_folders(
            data_dir=ds["data_dir"],
            output_path=output_path,
            dump_dir=ds["dump_dir"],
            device=ds["device"],
            meta_file=ds["meta_file"],
            max_folders=MAX_FOLDERS,
        )

    print("\n全部處理完成！")
