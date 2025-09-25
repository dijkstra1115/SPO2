import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def analyze_data_files():
    """
    分析 data 目錄中的所有 CSV 文件
    """
    data_dir = "data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print("=" * 80)
    print("RGB-SpO2 數據統計分析")
    print("=" * 80)
    
    # 總體統計
    total_files = len(csv_files)
    print(f"\n📁 數據文件總數: {total_files}")
    
    # 按文件類型分組
    file_groups = {}
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # 提取設備類型 (hm1, hm2, hm3)
        device_type = filename.split('-')[0]
        if device_type not in file_groups:
            file_groups[device_type] = []
        file_groups[device_type].append(filename)
    
    print(f"\n📊 按設備類型分組:")
    for device, files in file_groups.items():
        print(f"  {device}: {len(files)} 個文件")
        for file in sorted(files):
            print(f"    - {file}")
    
    # 詳細分析每個文件
    print(f"\n📈 詳細文件分析:")
    print("-" * 80)
    
    all_data = []
    total_rows = 0
    total_subjects = set()
    
    for file_path in sorted(csv_files):
        filename = os.path.basename(file_path)
        print(f"\n📄 文件: {filename}")
        
        try:
            # 讀取數據
            df = pd.read_csv(file_path)
            
            # 基本統計
            rows = len(df)
            cols = len(df.columns)
            total_rows += rows
            
            print(f"  📏 數據維度: {rows:,} 行 × {cols} 列")
            print(f"  📋 列名: {list(df.columns)}")
            
            # 檢查缺失值
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"  ⚠️  缺失值: {missing_values.to_dict()}")
            else:
                print(f"  ✅ 無缺失值")
            
            # 數值列統計
            numeric_cols = ['COLOR_R', 'COLOR_G', 'COLOR_B', 'SPO2']
            for col in numeric_cols:
                if col in df.columns:
                    print(f"  📊 {col}:")
                    print(f"    範圍: {df[col].min():.2f} - {df[col].max():.2f}")
                    print(f"    平均: {df[col].mean():.2f}")
                    print(f"    標準差: {df[col].std():.2f}")
            
            # 主體統計
            if 'Folder' in df.columns:
                unique_folders = df['Folder'].nunique()
                print(f"  👥 主體數量: {unique_folders}")
                
                # 提取主體ID
                subject_pattern = r'\[([^\]]+)\]'
                subject_ids = df['Folder'].str.extract(subject_pattern)[0].unique()
                total_subjects.update(subject_ids)
                
                print(f"  🆔 主體ID: {sorted(subject_ids)}")
            
            # SPO2 值分佈
            if 'SPO2' in df.columns:
                spo2_values = df['SPO2'].unique()
                spo2_range = f"{df['SPO2'].min()}-{df['SPO2'].max()}"
                print(f"  🩸 SPO2 範圍: {spo2_range}")
                print(f"  🩸 SPO2 唯一值: {sorted(spo2_values)}")
                
                # 詳細的 SPO2 分佈統計
                spo2_counts = df['SPO2'].value_counts().sort_index()
                print(f"  🩸 SPO2 分佈統計:")
                for spo2_val, count in spo2_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"    {spo2_val}%: {count:,} 次 ({percentage:.1f}%)")
            
            # 將數據添加到總體分析
            df['file_source'] = filename
            all_data.append(df)
            
        except Exception as e:
            print(f"  ❌ 讀取錯誤: {str(e)}")
    
    # 合併所有數據進行總體分析
    if all_data:
        print(f"\n🔍 總體數據分析:")
        print("-" * 80)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 總數據量: {len(combined_df):,} 行")
        print(f"👥 總主體數: {len(total_subjects)}")
        print(f"📁 總文件數: {total_files}")
        
        # 按設備類型統計
        print(f"\n📱 按設備類型統計:")
        for device, files in file_groups.items():
            device_data = combined_df[combined_df['file_source'].str.startswith(device)]
            if not device_data.empty:
                print(f"  {device}:")
                print(f"    數據量: {len(device_data):,} 行")
                subject_pattern = r'\[([^\]]+)\]'
                print(f"    主體數: {device_data['Folder'].str.extract(subject_pattern)[0].nunique()}")
                if 'SPO2' in device_data.columns:
                    print(f"    SPO2 範圍: {device_data['SPO2'].min()}-{device_data['SPO2'].max()}")
                    
                    # 設備類型的 SPO2 分佈
                    print(f"    SPO2 分佈:")
                    device_spo2_counts = device_data['SPO2'].value_counts().sort_index()
                    for spo2_val, count in device_spo2_counts.items():
                        percentage = (count / len(device_data)) * 100
                        print(f"      {spo2_val}%: {count:,} 次 ({percentage:.1f}%)")
        
        # 顏色通道統計
        print(f"\n🎨 顏色通道統計:")
        color_cols = ['COLOR_R', 'COLOR_G', 'COLOR_B']
        for col in color_cols:
            if col in combined_df.columns:
                print(f"  {col}:")
                print(f"    範圍: {combined_df[col].min():.2f} - {combined_df[col].max():.2f}")
                print(f"    平均: {combined_df[col].mean():.2f}")
                print(f"    標準差: {combined_df[col].std():.2f}")
        
        # SPO2 總體分佈
        if 'SPO2' in combined_df.columns:
            print(f"\n🩸 SPO2 總體分佈:")
            spo2_counts = combined_df['SPO2'].value_counts().sort_index()
            for spo2_val, count in spo2_counts.items():
                percentage = (count / len(combined_df)) * 100
                print(f"  {spo2_val}%: {count:,} 次 ({percentage:.1f}%)")
        
        # 主體分析
        print(f"\n👥 主體詳細分析:")
        subject_pattern = r'\[([^\]]+)\]'
        subject_data = combined_df['Folder'].str.extract(subject_pattern)[0]
        subject_counts = subject_data.value_counts().sort_index()
        print(f"  主體ID列表: {sorted(subject_counts.index.tolist())}")
        print(f"  每個主體的數據量:")
        for subject, count in subject_counts.items():
            print(f"    {subject}: {count:,} 行")
        
        # 數據質量檢查
        print(f"\n🔍 數據質量檢查:")
        print(f"  重複行數: {combined_df.duplicated().sum():,}")
        print(f"  完全重複的行: {combined_df.duplicated().sum():,}")
        
        # 檢查是否有異常值
        if 'SPO2' in combined_df.columns:
            normal_spo2 = (combined_df['SPO2'] >= 70) & (combined_df['SPO2'] <= 100)
            abnormal_count = (~normal_spo2).sum()
            if abnormal_count > 0:
                print(f"  ⚠️  異常 SPO2 值: {abnormal_count:,} 行")
                abnormal_values = combined_df[~normal_spo2]['SPO2'].unique()
                print(f"    異常值: {sorted(abnormal_values)}")
            else:
                print(f"  ✅ SPO2 值都在正常範圍內 (70-100%)")
    
    print(f"\n" + "=" * 80)
    print("統計分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    analyze_data_files()
