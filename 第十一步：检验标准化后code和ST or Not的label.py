import pandas as pd

original = pd.read_csv('final_model_data_quarterly.csv')
scaled = pd.read_csv('final_model_data_quarterly_scaled.csv')

# 检查两个文件中 ST or Not 是否一致
consistent = (original['ST or Not'] == scaled['ST or Not']).all()

print(f"标签一致性校验结果：{consistent}")