import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# 读取数据
data = pd.read_csv('final_model_data_quarterly_scaled.csv')

# 统计原始数据中0和1的数量
print("原始数据标签分布:")
print(data['ST or Not'].value_counts())

# 分离特征和标签
X = data.drop('ST or Not', axis=1)
y = data['ST or Not']

# 应用SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 合并处理后的数据
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['ST or Not'] = y_resampled

# 统计SMOTE后的标签分布
print("\nSMOTE后数据标签分布:")
print(resampled_data['ST or Not'].value_counts())

# 保存处理后的数据
resampled_data.to_csv('Ready to Model.csv', index=False)
print("\n处理后的数据已保存至：./Ready to Model.csv")