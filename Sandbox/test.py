import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

r2_df = pd.read_csv('fixed_combo_results_R2.csv')
mae_df = pd.read_csv('fixed_combo_results_MAE.csv')

r2_pivot = r2_df.set_index('train_file')
mae_pivot = mae_df.set_index('train_file')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
# R2 熱圖
sns.heatmap(r2_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0.0, ax=ax1, cbar_kws={'label': 'R2 Score'},
            annot_kws={'fontsize': 8})
ax1.set_title('R2 Score Heatmap')
ax1.set_xlabel('Test')
ax1.set_ylabel('Train')
# 旋轉 x 軸標籤 45 度
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

# MAE 熱圖 - 修復顯示問題
sns.heatmap(mae_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            center=10.0, ax=ax2, cbar_kws={'label': 'MAE'},
            annot_kws={'fontsize': 8})
ax2.set_title('MAE Heatmap')
ax2.set_xlabel('Test')
ax2.set_ylabel('Train')
# 旋轉 x 軸標籤 45 度
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()