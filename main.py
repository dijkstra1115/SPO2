import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def main():
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # CCT資料夾列表
    cct_folders = [
        'CCT_3200', 'CCT_3500', 'CCT_3800', 'CCT_4100', 
        'CCT_4400', 'CCT_4700', 'CCT_5000', 'CCT_5300', 'CCT_5600'
    ]
    
    # 通道列表
    channels = ['cg', 'cr', 'iq', 'dr', 'pos', 'chrom']
    
    # 為每個CCT資料夾分配顏色
    colors = plt.cm.Set1(np.linspace(0, 1, len(cct_folders)))
    
    # 存儲所有係數的字典，按通道分組
    all_coefficients = {channel: [] for channel in channels}
    
    print("開始處理數據...")
    
    # 遍歷每個CCT資料夾
    for i, cct_folder in enumerate(cct_folders):
        data_path = os.path.join('Data', 'CCT_Output_holding', cct_folder, 'data.csv')
        
        if not os.path.exists(data_path):
            print(f"警告: {data_path} 不存在，跳過")
            continue
            
        print(f"處理 {cct_folder}...")
        
        try:
            # 讀取數據
            df = pd.read_csv(data_path)
            print(f"  - 數據形狀: {df.shape}")
            print(f"  - Folder數量: {df['Folder'].nunique()}")
            print(f"  - Folder列表: {sorted(df['Folder'].unique())}")
            
            # 對每個Folder建立線性模型
            for folder in sorted(df['Folder'].unique()):
                folder_data = df[df['Folder'] == folder]
                
                if len(folder_data) < 10:  # 確保有足夠的數據點
                    print(f"    - Folder {folder}: 數據點不足，跳過")
                    continue
                
                # 對每個通道建立單獨的線性模型
                for channel in channels:
                    # 使用單個通道數據作為特徵，gt作為目標
                    features = folder_data[channel].values.reshape(-1, 1)
                    target = folder_data['gt'].values
                    
                    # 建立線性回歸模型
                    model = LinearRegression()
                    model.fit(features, target)
                    
                    # 提取係數 a (斜率) 和 b (截距)
                    a = model.coef_[0]  # 斜率
                    b = model.intercept_  # 截距
                    
                    # 儲存結果
                    all_coefficients[channel].append({
                        'cct_folder': cct_folder,
                        'folder': folder,
                        'a': a,  # 斜率
                        'b': b,  # 截距
                        'color': colors[i],
                        'color_index': i
                    })
                    
                    print(f"    - Folder {folder}, Channel {channel}: a(斜率)={a:.6f}, b(截距)={b:.6f}")
                
        except Exception as e:
            print(f"  錯誤處理 {cct_folder}: {str(e)}")
            continue
    
    # 為每個通道繪製散點圖
    for channel in channels:
        print(f"\n繪製 {channel} 通道的散點圖...")
        
        if not all_coefficients[channel]:
            print(f"  {channel} 通道沒有數據，跳過")
            continue
        
        plt.figure(figsize=(12, 8))
        
        # 按顏色分組繪製
        for i, cct_folder in enumerate(cct_folders):
            # 找出屬於當前CCT資料夾的係數
            folder_coeffs = [coeff for coeff in all_coefficients[channel] if coeff['color_index'] == i]
            
            if folder_coeffs:
                a_values = [coeff['a'] for coeff in folder_coeffs]
                b_values = [coeff['b'] for coeff in folder_coeffs]
                
                plt.scatter(a_values, b_values, 
                           c=[folder_coeffs[0]['color']], 
                           label=cct_folder, 
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
        
        plt.xlabel('Coefficient a (Slope)', fontsize=12)
        plt.ylabel('Coefficient b (Intercept)', fontsize=12)
        plt.title(f'{channel} channel: Scatter plot of linear model coefficients for different CCT values', fontsize=14)
        plt.legend(title='CCT values', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 添加統計信息
        channel_coeffs = all_coefficients[channel]
        plt.figtext(0.02, 0.02, f'Number of Models: {len(channel_coeffs)}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{channel}_coefficients_scatter_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  {channel} 通道散點圖已保存為: {channel}_coefficients_scatter_plot.png")
    
    # 顯示統計摘要
    print("\n=== 統計摘要 ===")
    for channel in channels:
        if all_coefficients[channel]:
            channel_coeffs = all_coefficients[channel]
            print(f"\n{channel} 通道:")
            print(f"  係數a(斜率)的範圍: [{min([c['a'] for c in channel_coeffs]):.6f}, {max([c['a'] for c in channel_coeffs]):.6f}]")
            print(f"  係數b(截距)的範圍: [{min([c['b'] for c in channel_coeffs]):.6f}, {max([c['b'] for c in channel_coeffs]):.6f}]")
            print(f"  係數a(斜率)的平均值: {np.mean([c['a'] for c in channel_coeffs]):.6f}")
            print(f"  係數b(截距)的平均值: {np.mean([c['b'] for c in channel_coeffs]):.6f}")

if __name__ == "__main__":
    main()
