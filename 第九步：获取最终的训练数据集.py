
import pandas as pd

# 1. 读取最终公司名单（code 列）
final_codes = pd.read_csv('final_company_codes.csv')

# 确保 code 是整数（和 ch 的一致）
final_codes['code'] = final_codes['code'].astype(int)

# 2. 读取原始季度数据
ch = pd.read_csv('merged_2019_IOM209_Data_0331.csv')

# 3. 筛选出用于建模的季度数据
model_data = ch[ch['code'].isin(final_codes['code'])]

# 4. 保存为新文件
model_data.to_csv('final_model_data_quarterly.csv', index=False)

print(f"筛选后的公司数量（唯一 code）：{model_data['code'].nunique()}")
print(f"总记录数（按季度展开）：{len(model_data)}")
