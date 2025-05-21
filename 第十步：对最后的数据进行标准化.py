from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. 读取原始数据
ch = pd.read_csv('final_model_data_quarterly.csv')

# 2. 指定要标准化的特征列
features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']

# 3. 初始化并拟合标准化器
scaler = StandardScaler()

# 4. 拷贝原始数据并进行标准化
ch_scaled = ch.copy()
ch_scaled[features] = scaler.fit_transform(ch[features])

# 5. 重排列顺序：code → 特征列 → ST or Not
ordered_columns = ['code'] + features + ['ST or Not']
ch_scaled = ch_scaled[ordered_columns]

# 6. 保存为新文件
ch_scaled.to_csv('final_model_data_quarterly_scaled.csv', index=False)
