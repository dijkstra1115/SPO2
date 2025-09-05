#!/usr/bin/env python3
"""
實驗執行腳本 - 從 config.json 讀取配置並執行訓練
"""

import json
import subprocess
import sys
import os
from pathlib import Path

def load_config(config_path="config.json"):
    """載入配置檔案"""
    if not os.path.exists(config_path):
        print(f"錯誤：找不到配置檔案 {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def build_command(config):
    """根據配置建立命令"""
    cmd = ["python", "train_loso_tcn.py"]
    
    # 預設值
    defaults = {
        "fps": 30,
        "seq_sec": 30,
        "stride_sec": 2,
        "epochs": 40,
        "patience": 10,
        "batch_size": 512,
        "hidden": 64,
        "levels": 3,
        "kernel": 5,
        "dropout": 0.1,
        "lr": 0.001,
        "seed": 42,
        "device": "cuda",
        "val_ratio": 0.1,
        "pos_win": 30,
        "rf_estimators": 10,
        "rf_max_depth": None,
        "svr_kernel": "rbf",
        "svr_C": 1.0,
        "svr_gamma": "scale"
    }
    
    # 基本參數
    cmd.extend(["--mode", config["mode"]])
    cmd.extend(["--model", config["model"]])
    
    # 訓練和測試檔案
    if config["mode"] == "fixed":
        cmd.extend(["--train_csv"] + config["train_csv"])
        cmd.extend(["--test_csv"] + config["test_csv"])
    elif config["mode"] == "loso":
        cmd.extend(["--csv", config["csv"]])
    
    # 模型參數（使用配置值或預設值）
    for param in ["fps", "seq_sec", "stride_sec", "epochs", "patience", "batch_size", 
                  "hidden", "levels", "kernel", "dropout", "lr", "seed", "device", "val_ratio"]:
        value = config.get(param, defaults[param])
        cmd.extend([f"--{param}", str(value)])
    
    # 布林參數
    if config.get("subject_normalization", False):
        cmd.append("--subject_normalization")
    if config.get("segment_normalization", False):
        cmd.append("--segment_normalization")
    if config.get("use_pos", False):
        cmd.append("--use_pos")
        pos_win = config.get("pos_win", defaults["pos_win"])
        cmd.extend(["--pos_win", str(pos_win)])
    
    # sklearn 模型參數
    if config["model"] == "rf":
        rf_estimators = config.get("rf_estimators", defaults["rf_estimators"])
        cmd.extend(["--rf_estimators", str(rf_estimators)])
        rf_max_depth = config.get("rf_max_depth", defaults["rf_max_depth"])
        if rf_max_depth is not None:
            cmd.extend(["--rf_max_depth", str(rf_max_depth)])
    elif config["model"] == "svr":
        svr_kernel = config.get("svr_kernel", defaults["svr_kernel"])
        svr_C = config.get("svr_C", defaults["svr_C"])
        svr_gamma = config.get("svr_gamma", defaults["svr_gamma"])
        cmd.extend(["--svr_kernel", svr_kernel])
        cmd.extend(["--svr_C", str(svr_C)])
        cmd.extend(["--svr_gamma", str(svr_gamma)])
    
    return cmd

def main():
    """主函數"""
    # 檢查是否在正確的目錄
    if not os.path.exists("train_loso_tcn.py"):
        print("錯誤：請在包含 train_loso_tcn.py 的目錄中執行此腳本")
        sys.exit(1)
    
    # 載入配置
    config = load_config()
    
    # 建立命令
    cmd = build_command(config)
    
    # 顯示將要執行的命令
    print("=" * 60)
    print("實驗配置：")
    print(f"模式: {config['mode']}")
    print(f"模型: {config['model']}")
    if config["mode"] == "fixed":
        print(f"訓練集: {len(config['train_csv'])} 個檔案")
        print(f"測試集: {len(config['test_csv'])} 個檔案")
        print(f"有效組合數: {len(config['train_csv']) * len(config['test_csv']) - len(config['train_csv'])}")
    print("=" * 60)
    print("執行命令:")
    print(" ".join(cmd))
    print("=" * 60)
    
    # 確認執行
    response = input("是否執行？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("已取消執行")
        sys.exit(0)
    
    # 執行命令
    try:
        result = subprocess.run(cmd, check=True)
        print("\n實驗完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n執行失敗，錯誤代碼: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n用戶中斷執行")
        sys.exit(1)

if __name__ == "__main__":
    main()
