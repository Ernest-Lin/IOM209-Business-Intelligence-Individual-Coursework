import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Step 1: 读取数据
df = pd.read_csv("final_model_data_quarterly_scaled.csv")

# Step 2: 添加季度时间标签（每家公司 20 条记录，从 2015Q1 到 2019Q4）
quarters = pd.date_range(start="2015-03-31", periods=20, freq="Q")
df['date'] = np.tile(quarters, len(df) // 20)

# Step 3: 构造滞后变量
target_col = 'ST or Not'
exclude_cols = ['code', 'date', target_col]
feature_cols = [col for col in df.columns if col not in exclude_cols]

# 为每个特征添加 t-1 和 t-2 滞后项（按 code 分组）
for col in feature_cols:
    df[f'{col}_lag1'] = df.groupby('code')[col].shift(1)
    df[f'{col}_lag2'] = df.groupby('code')[col].shift(2)

# Step 4: 删除滞后引入的缺失行
df_clean = df.dropna().reset_index(drop=True)

# Step 5: 拆分特征和标签
X = df_clean.drop(columns=['code', 'date', target_col])
y = df_clean[target_col]

# 备份原始样本的 code 和 date
meta_info = df_clean[['code', 'date']].reset_index(drop=True)

# Step 6: 使用 SMOTE 做样本均衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 7: 构建完整 DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled[target_col] = y_resampled

# Step 8: 填回 code 和 date（新增样本填充为 NaN）
original_len = len(meta_info)
num_new_samples = len(df_resampled) - original_len

code_extended = pd.concat([
    meta_info['code'],
    pd.Series([np.nan] * num_new_samples, name='code')
], ignore_index=True)

date_extended = pd.concat([
    meta_info['date'],
    pd.Series([pd.NaT] * num_new_samples, name='date')
], ignore_index=True)

df_resampled['code'] = code_extended
df_resampled['date'] = date_extended

# Step 9: 调整列顺序（可选）
ordered_cols = ['code', 'date'] + [col for col in df_resampled.columns if col not in ['code', 'date']]
df_resampled = df_resampled[ordered_cols]

# Step 10: 保存结果
df_resampled.to_csv("Individual Ready to Model.csv", index=False)

