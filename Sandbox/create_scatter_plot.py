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

def create_scatter_plot(predictions_file, output_file=None, combined=True):
    """
    創建 true label vs prediction 散點圖
    
    Args:
        predictions_file: 預測結果CSV文件路徑
        output_file: 輸出圖片文件路徑（可選）
        combined: 是否創建組合圖（預設True）
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
    
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    test_files = df['test_file'].unique()
    print(f"將為 {len(test_files)} 個測試文件創建散點圖")
    
    if combined:
        # 創建組合圖
        create_combined_plot(df, test_files, output_file)
    else:
        # 為每個測試文件創建單獨的散點圖
        create_individual_plots(df, test_files, output_file)
    
    # 顯示整體統計
    print(f"\n=== 整體統計摘要 ===")
    overall_r2 = r2_score(df['true_label'], df['prediction'])
    overall_mae = mean_absolute_error(df['true_label'], df['prediction'])
    overall_rmse = np.sqrt(mean_squared_error(df['true_label'], df['prediction']))
    print(f"總樣本數: {len(df)}")
    print(f"整體 R² 分數: {overall_r2:.3f}")
    print(f"整體平均絕對誤差 (MAE): {overall_mae:.2f}")
    print(f"整體均方根誤差 (RMSE): {overall_rmse:.2f}")

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
    parser = argparse.ArgumentParser(description='創建 true label vs prediction 散點圖')
    parser.add_argument('--predictions_file', type=str, default='predictions_results.csv',
                        help='預測結果CSV文件路徑 (預設: predictions_results.csv)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='輸出圖片文件路徑 (預設: combined_scatter_plots.png)')
    parser.add_argument('--combined', action='store_true', default=True,
                        help='創建組合圖（預設）')
    parser.add_argument('--individual', action='store_true',
                        help='創建單獨的散點圖')
    
    args = parser.parse_args()
    
    # 如果指定了 individual，則設置 combined 為 False
    if args.individual:
        combined = False
    else:
        combined = args.combined
    
    create_scatter_plot(args.predictions_file, args.output_file, combined)

if __name__ == "__main__":
    main()
