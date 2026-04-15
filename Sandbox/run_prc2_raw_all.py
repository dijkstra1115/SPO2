import os, json, joblib, pandas as pd, common
from common import ensemble_predict_and_evaluate, filter_feat_df_by_spo2_range, load_features_from_csv_paths
with open('./output/model/model_pool_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
raw_pool = joblib.load('./output/model/model_pool_raw.joblib')
common.GLOBAL_MODEL_BLEND = 0.35
segment_length = config['SEGMENT_LENGTH']
use_normalization = config['USE_NORMALIZATION']
rows=[]
for csv_path in ['./data_new/prc2-c930.csv','./data_new/prc2-i16.csv','./data_new/prc2-i16m.csv']:
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    feat_df = load_features_from_csv_paths([csv_path], verbose=False)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=segment_length, verbose=False)
    results_df = ensemble_predict_and_evaluate(raw_pool, feat_df, None, 20, segment_length, use_normalization, False, common.OUTPUT_DIR)
    rows.append({'csv_name': csv_name, 'mean_R2': results_df['R2'].mean(), 'mean_PCC': results_df['PCC'].mean(), 'overall_PCC': results_df['overall_PCC'].iloc[0], 'mean_MAE': results_df['MAE'].mean(), 'n_subjects': len(results_df)})
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv('./output/prc2_raw_all_topk20.csv', index=False, encoding='utf-8')
