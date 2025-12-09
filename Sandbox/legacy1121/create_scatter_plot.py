#!/usr/bin/env python3
"""
創建 true label vs prediction 散點圖
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import argparse
import os

def create_scatter_plot(predictions_file, output_file=None, combined=True, train_file=None, test_file=None):
    """
    創建 true label vs prediction 散點圖
    
    Args:
        predictions_file: 預測結果CSV文件路徑
        output_file: 輸出圖片文件路徑（可選）
        combined: 是否創建組合圖（預設True）
        train_file: 指定要過濾的訓練文件名列表（可選）
        test_file: 指定要過濾的測試文件名列表（可選）
    """
    # 讀取預測結果
    if not os.path.exists(predictions_file):
        print(f"錯誤：找不到文件 {predictions_file}")
        return
    
    df = pd.read_csv(predictions_file)

    # 移除指定的 test_file
    df = df[df['test_file'] != "hm1-s9280.csv"]
    
    # 檢查必要的列
    required_cols = ['true_label', 'prediction']
    if not all(col in df.columns for col in required_cols):
        print(f"錯誤：文件必須包含列 {required_cols}")
        return
    
    # 檢查是否有 train_file 列
    if 'train_file' not in df.columns:
        print("警告：數據中沒有 'train_file' 列，將只按測試文件分組")
        train_file = None
    
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建交叉配對（排除自己與自己的配對）
    if train_file is not None and test_file is not None:
        # 確保輸入是列表
        if not isinstance(train_file, list):
            train_file = [train_file]
        if not isinstance(test_file, list):
            test_file = [test_file]
        
        # 創建交叉配對，排除自己與自己的配對
        train_test_pairs = []
        for train_f in train_file:
            for test_f in test_file:
                if train_f != test_f:  # 排除自己與自己的配對
                    train_test_pairs.append((train_f, test_f))
        
        if len(train_test_pairs) == 0:
            print("錯誤：沒有有效的交叉配對（所有配對都是自己與自己）")
            return
        
        print(f"將為 {len(train_test_pairs)} 個交叉配對創建散點圖")
        print("配對列表:", train_test_pairs)
        
        # 過濾數據以只包含這些配對
        pair_mask = df.apply(lambda row: (row['train_file'], row['test_file']) in train_test_pairs, axis=1)
        df = df[pair_mask]
        
        # 檢查是否有數據
        if len(df) == 0:
            print("沒有找到任何配對的數據，跳過繪圖")
            return
        
        # 創建散點圖
        if combined:
            create_combined_plot_by_pairs(df, train_test_pairs, output_file)
        else:
            create_individual_plots_by_pairs(df, train_test_pairs, output_file)
    
    else:
        # 如果沒有指定 train_file 或 test_file，使用原來的邏輯（按測試文件分組）
        test_files = df['test_file'].unique()
        print(f"將為 {len(test_files)} 個測試文件創建散點圖")
        
        if combined:
            create_combined_plot(df, test_files, output_file)
        else:
            create_individual_plots(df, test_files, output_file)
    
    # 顯示整體統計
    print(f"\n=== 整體統計摘要 ===")
    if len(df) > 0:
        overall_r2 = r2_score(df['true_label'], df['prediction'])
        overall_mae = mean_absolute_error(df['true_label'], df['prediction'])
        overall_rmse = np.sqrt(mean_squared_error(df['true_label'], df['prediction']))
        print(f"總樣本數: {len(df)}")
        print(f"整體 R² 分數: {overall_r2:.3f}")
        print(f"整體平均絕對誤差 (MAE): {overall_mae:.2f}")
        print(f"整體均方根誤差 (RMSE): {overall_rmse:.2f}")
    else:
        print("沒有找到符合條件的數據，程式結束")
        return

def create_combined_plot(df, test_files, output_file):
    """創建組合散點圖"""
    n_files = len(test_files)
    
    # 計算合適的網格布局
    if n_files <= 1:
        rows, cols = 1, 1
    elif n_files <= 2:
        rows, cols = 1, 2
    elif n_files <= 4:
        rows, cols = 2, 2
    elif n_files <= 6:
        rows, cols = 2, 3
    elif n_files <= 9:
        rows, cols = 3, 3
    elif n_files <= 12:
        rows, cols = 3, 4
    else:
        # 對於更多文件，使用更緊湊的布局
        cols = int(np.ceil(np.sqrt(n_files)))
        rows = int(np.ceil(n_files / cols))
    
    # 創建子圖
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # 確保 axes 是二維數組
    if n_files == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 為每個測試文件創建子圖
    for i, test_file in enumerate(test_files):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        test_data = df[df['test_file'] == test_file]
        
        # 繪製散點圖
        ax.scatter(test_data['true_label'], test_data['prediction'], alpha=0.6, s=20)
        
        # 添加完美預測線 (y=x)
        min_val = min(test_data['true_label'].min(), test_data['prediction'].min())
        max_val = max(test_data['true_label'].max(), test_data['prediction'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y=x')
        
        # 計算評估指標
        r2 = r2_score(test_data['true_label'], test_data['prediction'])
        mae = mean_absolute_error(test_data['true_label'], test_data['prediction'])
        rmse = np.sqrt(mean_squared_error(test_data['true_label'], test_data['prediction']))
        
        ax.set_xlabel('True Label (SpO2)')
        ax.set_ylabel('Prediction (SpO2)')
        ax.set_title(f'{test_file.replace(".csv", "")}\nR²={r2:.3f}, MAE={mae:.2f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 顯示統計信息
        print(f"\n=== {test_file} 統計摘要 ===")
        print(f"樣本數: {len(test_data)}")
        print(f"R² 分數: {r2:.3f}")
        print(f"平均絕對誤差 (MAE): {mae:.2f}")
        print(f"均方根誤差 (RMSE): {rmse:.2f}")
    
    # 隱藏多餘的子圖
    for i in range(n_files, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    # 保存組合圖
    if output_file is None:
        combined_output_file = 'combined_scatter_plots.png'
    else:
        combined_output_file = output_file
    
    plt.tight_layout()
    plt.savefig(combined_output_file, dpi=300, bbox_inches='tight')
    print(f"\n組合散點圖已保存為: {combined_output_file}")
    plt.show()

def create_combined_plot_by_pairs(df, train_test_pairs, output_file):
    """創建按 train-test 配對分組的組合散點圖"""
    n_pairs = len(train_test_pairs)
    
    # 計算合適的網格布局
    if n_pairs <= 1:
        rows, cols = 1, 1
    elif n_pairs <= 2:
        rows, cols = 1, 2
    elif n_pairs <= 4:
        rows, cols = 2, 2
    elif n_pairs <= 6:
        rows, cols = 2, 3
    elif n_pairs <= 9:
        rows, cols = 3, 3
    elif n_pairs <= 12:
        rows, cols = 3, 4
    else:
        # 對於更多配對，使用更緊湊的布局
        cols = int(np.ceil(np.sqrt(n_pairs)))
        rows = int(np.ceil(n_pairs / cols))
    
    # 創建子圖
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # 確保 axes 是二維數組
    if n_pairs == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 為每個 train-test 配對創建子圖
    for i, (train_file, test_file) in enumerate(train_test_pairs):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # 過濾特定配對的數據
        if 'train_file' in df.columns:
            pair_data = df[(df['train_file'] == train_file) & (df['test_file'] == test_file)]
        else:
            pair_data = df[df['test_file'] == test_file]
        
        if len(pair_data) == 0:
            ax.text(0.5, 0.5, f'無數據\n{train_file}\n→ {test_file}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{train_file}\n→ {test_file}', fontsize=10)
            continue
        
        # 繪製散點圖
        ax.scatter(pair_data['true_label'], pair_data['prediction'], alpha=0.6, s=20)
        
        # 添加完美預測線 (y=x)
        min_val = min(pair_data['true_label'].min(), pair_data['prediction'].min())
        max_val = max(pair_data['true_label'].max(), pair_data['prediction'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y=x')
        
        # 計算評估指標
        r2 = r2_score(pair_data['true_label'], pair_data['prediction'])
        mae = mean_absolute_error(pair_data['true_label'], pair_data['prediction'])
        rmse = np.sqrt(mean_squared_error(pair_data['true_label'], pair_data['prediction']))
        
        ax.set_xlabel('True Label (SpO2)')
        ax.set_ylabel('Prediction (SpO2)')
        ax.set_title(f'{train_file}\n→ {test_file}\nR²={r2:.3f}, MAE={mae:.2f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 顯示統計信息
        print(f"\n=== {train_file} → {test_file} 統計摘要 ===")
        print(f"樣本數: {len(pair_data)}")
        print(f"R² 分數: {r2:.3f}")
        print(f"平均絕對誤差 (MAE): {mae:.2f}")
        print(f"均方根誤差 (RMSE): {rmse:.2f}")
    
    # 隱藏多餘的子圖
    for i in range(n_pairs, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    # 保存組合圖
    if output_file is None:
        combined_output_file = 'combined_scatter_plots_by_pairs.png'
    else:
        combined_output_file = output_file
    
    plt.tight_layout()
    plt.savefig(combined_output_file, dpi=300, bbox_inches='tight')
    print(f"\n組合散點圖已保存為: {combined_output_file}")
    plt.show()

def create_individual_plots_by_pairs(df, train_test_pairs, output_file):
    """為每個 train-test 配對創建單獨的散點圖"""
    for i, (train_file, test_file) in enumerate(train_test_pairs):
        # 過濾特定配對的數據
        if 'train_file' in df.columns:
            pair_data = df[(df['train_file'] == train_file) & (df['test_file'] == test_file)]
        else:
            pair_data = df[df['test_file'] == test_file]
        
        if len(pair_data) == 0:
            print(f"警告：配對 {train_file} → {test_file} 沒有數據")
            continue
        
        # 創建單個散點圖
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 繪製散點圖
        ax.scatter(pair_data['true_label'], pair_data['prediction'], alpha=0.6, s=30)
        
        # 添加完美預測線 (y=x)
        min_val = min(pair_data['true_label'].min(), pair_data['prediction'].min())
        max_val = max(pair_data['true_label'].max(), pair_data['prediction'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction line (y=x)')
        
        # 計算評估指標
        r2 = r2_score(pair_data['true_label'], pair_data['prediction'])
        mae = mean_absolute_error(pair_data['true_label'], pair_data['prediction'])
        rmse = np.sqrt(mean_squared_error(pair_data['true_label'], pair_data['prediction']))
        
        ax.set_xlabel('True Label (SpO2)')
        ax.set_ylabel('Prediction (SpO2)')
        ax.set_title(f'Train: {train_file} → Test: {test_file}\nR² = {r2:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存圖片
        if output_file is None:
            pair_output_file = f'scatter_plot_{train_file.replace(".csv", "")}_to_{test_file.replace(".csv", "")}.png'
        else:
            base_name = output_file.replace('.png', '')
            pair_output_file = f'{base_name}_{train_file.replace(".csv", "")}_to_{test_file.replace(".csv", "")}.png'
        
        plt.tight_layout()
        plt.savefig(pair_output_file, dpi=300, bbox_inches='tight')
        print(f"散點圖已保存為: {pair_output_file}")
        
        # 顯示統計信息
        print(f"\n=== {train_file} → {test_file} 統計摘要 ===")
        print(f"樣本數: {len(pair_data)}")
        print(f"R² 分數: {r2:.3f}")
        print(f"平均絕對誤差 (MAE): {mae:.2f}")
        print(f"均方根誤差 (RMSE): {rmse:.2f}")
        
        plt.show()

def create_individual_plots(df, test_files, output_file):
    """為每個測試文件創建單獨的散點圖"""
    for i, test_file in enumerate(test_files):
        test_data = df[df['test_file'] == test_file]
        
        # 創建單個散點圖
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 繪製散點圖
        ax.scatter(test_data['true_label'], test_data['prediction'], alpha=0.6, s=30)
        
        # 添加完美預測線 (y=x)
        min_val = min(test_data['true_label'].min(), test_data['prediction'].min())
        max_val = max(test_data['true_label'].max(), test_data['prediction'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction line (y=x)')
        
        # 計算評估指標
        r2 = r2_score(test_data['true_label'], test_data['prediction'])
        mae = mean_absolute_error(test_data['true_label'], test_data['prediction'])
        rmse = np.sqrt(mean_squared_error(test_data['true_label'], test_data['prediction']))
        
        ax.set_xlabel('True Label (SpO2)')
        ax.set_ylabel('Prediction (SpO2)')
        ax.set_title(f'Test File: {test_file}\nR² = {r2:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存圖片
        if output_file is None:
            test_output_file = f'scatter_plot_{test_file.replace(".csv", "")}.png'
        else:
            base_name = output_file.replace('.png', '')
            test_output_file = f'{base_name}_{test_file.replace(".csv", "")}.png'
        
        plt.tight_layout()
        plt.savefig(test_output_file, dpi=300, bbox_inches='tight')
        print(f"散點圖已保存為: {test_output_file}")
        
        # 顯示統計信息
        print(f"\n=== {test_file} 統計摘要 ===")
        print(f"樣本數: {len(test_data)}")
        print(f"R² 分數: {r2:.3f}")
        print(f"平均絕對誤差 (MAE): {mae:.2f}")
        print(f"均方根誤差 (RMSE): {rmse:.2f}")
        
        plt.show()

def main():
    # ===== 在這裡直接設定參數 =====
    predictions_file = 'predictions_results.csv'  # 預測結果文件路徑
    output_file = None  # 輸出圖片文件路徑（None 表示使用預設名稱）
    combined = True  # True: 創建組合圖, False: 創建單獨的散點圖
    
    # 支援字符串或列表格式
    train_file = ["prc-c920.csv", "prc-i15.csv", "prc-i15m.csv", "prc-s9160.csv", "prc-s9180.csv"]  # 指定要過濾的訓練文件名（可以是字符串或列表，None 表示不過濾）
    test_file = ["prc-c920.csv", "prc-i15.csv", "prc-i15m.csv", "prc-s9160.csv", "prc-s9180.csv"]  # 指定要過濾的測試文件名（可以是字符串或列表，None 表示不過濾）
    # ================================
    
    create_scatter_plot(predictions_file, output_file, combined, train_file, test_file)

if __name__ == "__main__":
    main()
