import pandas as pd
import numpy as np
import os
from typing import List, Optional

DATA_DIR = "//169.254.2.1/Algorithm/HehuanMountain3_SPO2"
DEVICE = "SM-S9380-Front"
META_FILE = "Meta.csv"
SPO2_MIN = 70
SPO2_MAX = 100
IQA_MOTION_THRESHOLD = 0.9

def process_folder_data(folder_name: str, data_dir: str) -> Optional[pd.DataFrame]:
    """
    處理單一 Folder 的資料，合併 {DEVICE}.csv 和 masimo.csv
    
    過濾條件：
    1. SPO2 資料內插成與 dump 相同的單位（偵）
    2. IQA Motion < 0.9 的偵及後續所有偵都會被過濾掉
    3. 只保留 SPO2 在 70-99 範圍內的資料
    
    Args:
        folder_name: Folder 名稱
        data_dir: 資料根目錄
        
    Returns:
        合併後的 DataFrame 或 None (如果處理失敗)
    """
    try:
        # 讀取 {DEVICE}.csv (RGB 資料 + IQA Motion)
        dump_path = os.path.join(data_dir, "Dump", folder_name, f"{DEVICE}.csv")
        dump_data = pd.read_csv(dump_path, usecols=["COLOR_R", "COLOR_G", "COLOR_B", "IQA Motion"])
        print(f"  - {DEVICE}.csv: {len(dump_data)} 偵")
        
        # 讀取 masimo.csv (SPO2 資料)
        masimo_path = os.path.join(data_dir, "Label", folder_name, "masimo.csv")
        masimo_data = pd.read_csv(masimo_path, usecols=["SPO2"])
        print(f"  - masimo.csv: {len(masimo_data)} 秒")
        
        # 處理 SPO2 缺失值 (將 "--" 等非數值轉換為 NaN)
        masimo_data["SPO2"] = pd.to_numeric(masimo_data["SPO2"], errors='coerce')
        missing_count = masimo_data["SPO2"].isna().sum()
        if missing_count > 0:
            print(f"  - SPO2 缺失值: {missing_count}/{len(masimo_data)} 秒")
        
        # 檢查 SPO2 資料是否有效
        if masimo_data["SPO2"].isna().all():
            print(f"  - 警告: {folder_name} 的 SPO2 資料全部缺失")
            return None
        
        # 檢查是否有 SPO2 在指定範圍內的數據
        spo2_in_range = (masimo_data["SPO2"].dropna() >= SPO2_MIN) & (masimo_data["SPO2"].dropna() <= SPO2_MAX)
        if not spo2_in_range.any():
            print(f"  - 跳過 {folder_name}: 沒有 SPO2 在 {SPO2_MIN}-{SPO2_MAX} 範圍內的數據")
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
        
        # 將 SPO2 資料擴展到每偵 (每個 SPO2 值重複30次)
        spo2_expanded = np.repeat(masimo_data["SPO2"].values, 30)
        
        # 找出 SPO2 缺失的位置
        spo2_valid_mask = ~np.isnan(spo2_expanded)
        
        # 找出 IQA Motion < 閾值的位置（該偵及後續所有偵都要過濾掉）
        iqa_motion = dump_data["IQA Motion"].values
        iqa_invalid_mask = iqa_motion < IQA_MOTION_THRESHOLD
        
        # 找到第一個 IQA Motion < 閾值的位置
        first_invalid_idx = np.argmax(iqa_invalid_mask) if iqa_invalid_mask.any() else len(iqa_invalid_mask)
        
        # 如果找到無效位置，則該位置及之後的所有資料都過濾掉
        if first_invalid_idx < len(iqa_invalid_mask):
            print(f"  - IQA Motion < {IQA_MOTION_THRESHOLD} 從第 {first_invalid_idx} 偵開始，過濾掉 {len(iqa_invalid_mask) - first_invalid_idx} 偵")
            # 創建 IQA Motion 有效遮罩（只保留無效位置之前的資料）
            iqa_valid_mask = np.zeros(len(iqa_invalid_mask), dtype=bool)
            iqa_valid_mask[:first_invalid_idx] = True
        else:
            iqa_valid_mask = np.ones(len(iqa_invalid_mask), dtype=bool)
            print(f"  - IQA Motion 全部有效")
        
        # 找出 SPO2 在指定範圍內的位置
        spo2_range_mask = (spo2_expanded >= SPO2_MIN) & (spo2_expanded <= SPO2_MAX)
        
        # 結合所有過濾條件：SPO2 非缺失、IQA Motion 有效、SPO2 在指定範圍內
        valid_mask = spo2_valid_mask & iqa_valid_mask & spo2_range_mask

        # 只要有數據不符合條件，則過濾掉整個 Folder
        if not valid_mask.all():
            print(f"  - 警告: {folder_name} 過濾後無有效資料")
            return None
        
        # 過濾掉無效的資料
        dump_filtered = dump_data[valid_mask]
        spo2_filtered = spo2_expanded[valid_mask]
        
        print(f"  - 過濾後有效資料: {len(dump_filtered)} 偵")
        
        if len(dump_filtered) == 0:
            print(f"  - 警告: {folder_name} 過濾後無有效資料")
            return None
        
        # 建立最終的 DataFrame
        result_df = pd.DataFrame({
            'Folder': [folder_name] * len(dump_filtered),
            'COLOR_R': dump_filtered["COLOR_R"].values,
            'COLOR_G': dump_filtered["COLOR_G"].values,
            'COLOR_B': dump_filtered["COLOR_B"].values,
            'SPO2': spo2_filtered
        })
        
        return result_df
        
    except FileNotFoundError as e:
        print(f"  - 錯誤: 找不到檔案 {e}")
        return None
    except Exception as e:
        print(f"  - 錯誤: 處理 {folder_name} 時發生異常: {e}")
        return None

def merge_all_folders(data_dir: str, output_path: str = "data.csv", max_folders: Optional[int] = None) -> None:
    """
    處理所有 Folder 並合併成一個大的資料集
    
    Args:
        data_dir: 資料根目錄
        output_path: 輸出檔案路徑
        max_folders: 最大處理的 Folder 數量，None 表示處理全部
    """
    # 讀取 Meta.csv
    meta_path = os.path.join(data_dir, META_FILE)
    
    try:
        meta_df = pd.read_csv(meta_path)
        print(f"讀取 {META_FILE}: {len(meta_df)} 個 Folder")
    except FileNotFoundError:
        print(f"錯誤: 找不到 {meta_path}")
        return
    
    if "Folder" not in meta_df.columns:
        print(f"錯誤: {META_FILE} 中沒有 'Folder' 欄位")
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
        
        folder_data = process_folder_data(folder, data_dir)
        
        if folder_data is not None:
            all_data.append(folder_data)
            successful_folders += 1
            print(f"  - 成功處理，資料筆數: {len(folder_data)}")
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
        
        # 顯示每個 Folder 的資料統計
        folder_stats = final_df.groupby('Folder').size()
        print(f"  - 每個 Folder 的資料筆數:")
        for folder, count in folder_stats.items():
            print(f"    {folder}: {count} 筆")
        
        # 輸出到檔案
        final_df.to_csv(output_path, index=False)
        print(f"\n資料已保存到: {output_path}")
        
    else:
        print("\n錯誤: 沒有成功處理任何 Folder")

if __name__ == "__main__":
    print("開始處理 PRC SpO2 資料...")
    
    # 設定最大處理的 Folder 數量 (None 表示處理全部)
    MAX_FOLDERS = None  # 可以改為數字，例如: MAX_FOLDERS = 5
    
    merge_all_folders(DATA_DIR, max_folders=MAX_FOLDERS)
    print("處理完成！")