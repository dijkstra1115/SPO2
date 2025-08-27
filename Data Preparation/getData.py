import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d


DATA_DIR = "//169.254.2.1/Public/SpO2/SpO2_illumination"


def getsixchannels(rgb):

    yiq_i =  0.595716 * rgb[0, :] - 0.274453 * rgb[1, :] - 0.321263 * rgb[2, :] 
    ycgcr_cg = 128.0 -  81.085 * rgb[0, :] / 255.0 + 112.000 * rgb[1, :] / 255.0 - 30.915 * rgb[2, :] / 255.0
    ycgcr_cr = 128.0 + 112.000 * rgb[0, :] / 255.0 -  93.786 * rgb[1, :] / 255.0 - 18.214 * rgb[2, :] / 255.0
    ydbdr_dr = -1.333 * rgb[0, :] + 1.116 * rgb[1, :] + 0.217 * rgb[2, :] 
    pos_y = -2*rgb[0, :] + rgb[1, :] + rgb[2, :]
    chrom_x =   3*rgb[0, :] - 2*rgb[1, :]

    ret = [ycgcr_cg, ycgcr_cr, yiq_i, ydbdr_dr, pos_y, chrom_x]

    return ret


def getCCT(rgb_):

    rgb = rgb_ / 255.0

    X = 0.4124*rgb[0, :] + 0.3576*rgb[1, :] + 0.1805*rgb[2, :]
    Y = 0.2126*rgb[0, :] + 0.7152*rgb[1, :] + 0.0722*rgb[2, :]
    Z = 0.0193*rgb[0, :] + 0.1192*rgb[1, :] + 0.9505*rgb[2, :]

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    n = (x-0.3320)/(0.1858-y)
    CCT = 437*pow(n, 3) + 3601*pow(n, 2) + 6861*n + 5517

    return CCT


def interpLabels(label, n, kind='linear'):
    if (len(label) == n):
        return label
    interpolator = interp1d(np.linspace(0, 1, len(label)), label, kind=kind)
    return interpolator(np.linspace(0, 1, n))


def get_cct_folder_name(times):
    """根據Times值返回對應的CCT資料夾名稱"""
    cct_value = 5600 - (times - 10) * 300
    return f"CCT_{cct_value}"


if __name__ == "__main__":
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta0626.csv"))
    
    # 創建輸出目錄
    output_base_dir = "CCT_Output"
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # 為每個Times值創建對應的CCT資料夾
    for times in range(10, 19):
        cct_folder = get_cct_folder_name(times)
        cct_path = os.path.join(output_base_dir, cct_folder)
        if not os.path.exists(cct_path):
            os.makedirs(cct_path)
        print(f"創建資料夾: {cct_path}")
    
    # 處理每個meta記錄
    for i in range(len(meta)):
        meta_tmp = meta.iloc[i]
        times = meta_tmp["Times"]
        
        # 只處理Times 1-9的數據
        if times <= 9 or times > 18:
            continue
            
        print(f"處理 Times={times}, Folder={meta_tmp['Folder']}")
        
        # 讀取RGB數據
        data = pd.read_csv(os.path.join(DATA_DIR, "Dump", meta_tmp["Folder"], "C930.csv"))
        
        # 讀取真實值並插值
        try:
            gt = pd.read_csv(os.path.join(DATA_DIR, "Label", meta_tmp["Folder"], "masimo.csv"), usecols=["SPO2"])
            gt_np = gt.values.squeeze(-1)
            if type(gt_np[0]) is not np.int64:
                print(f"gt 缺少數據: {meta_tmp['Folder']}")
                continue
        except:
            print(f"找不到 {meta_tmp['Folder']} 的 masimo.csv")
            continue
        
        gt_interpolated = interpLabels(gt["SPO2"].values, len(data), kind='previous')
        
        # 獲取六個通道數據
        rgb = np.vstack((data["COLOR_R"], data["COLOR_G"], data["COLOR_B"]))
        cg, cr, iq, dr, pos, chrom = getsixchannels(rgb)
        
        # 創建輸出DataFrame
        output_df = pd.DataFrame({
            'Folder': [meta_tmp["Folder"]] * len(data),
            'COLOR_R': data["COLOR_R"].values,
            'COLOR_G': data["COLOR_G"].values,
            'COLOR_B': data["COLOR_B"].values,
            'cg': cg,
            'cr': cr,
            'iq': iq,
            'dr': dr,
            'pos': pos,
            'chrom': chrom,
            'gt': gt_interpolated
        })
        
        # 保存到對應的CCT資料夾
        cct_folder = get_cct_folder_name(times)
        output_path = os.path.join(output_base_dir, cct_folder, "data.csv")
        
        # 如果文件已存在，追加數據；否則創建新文件
        if os.path.exists(output_path):
            output_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            output_df.to_csv(output_path, index=False)
            
        print(f"已保存到: {output_path}")
    
    print("所有數據處理完成！")
    
    # 顯示每個CCT資料夾的數據統計
    for times in range(10, 19):
        cct_folder = get_cct_folder_name(times)
        output_path = os.path.join(output_base_dir, cct_folder, "data.csv")
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            print(f"{cct_folder}: {len(df)} 行數據")