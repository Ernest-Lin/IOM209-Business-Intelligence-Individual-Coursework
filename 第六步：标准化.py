import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

agg_ch_cleaned = pd.read_csv('merged_aftergroup_cleaned_0331.csv')

# 财务特征列（跟之前一致）
features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']

# 创建 StandardScaler 实例
scaler = StandardScaler()

# 对财务特征进行 Z-score 标准化
X_scaled = scaler.fit_transform(agg_ch_cleaned[features])

# 转换成标准化后的 DataFrame（保留 code 和 ST_flag）
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
X_scaled_df['code'] = agg_ch_cleaned['code'].values
X_scaled_df['ST_flag'] = agg_ch_cleaned['ST_flag'].values

# 重新排列列顺序，把 code 放到第一列
cols = ['code'] + [col for col in X_scaled_df.columns if col != 'code']
X_scaled_df = X_scaled_df[cols]

X_scaled_df.to_csv('X_scaled_0331.csv', index=False)

