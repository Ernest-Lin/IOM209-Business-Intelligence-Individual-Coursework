import pandas as pd
import numpy as np

matched_df = pd.read_csv('KNN_Code.csv')

# 获取所有匹配到的唯一非 ST 公司 code
matched_nonst_codes = matched_df['Matched_nonST_code'].unique()

# 获取所有 ST 公司 code
st_codes = matched_df['ST_code'].unique()

# 合并成用于建模的公司名单
final_company_codes = np.concatenate([st_codes, matched_nonst_codes])
final_company_codes_df = pd.DataFrame(final_company_codes, columns=['code'])
final_company_codes_df.to_csv('final_company_codes.csv', index=False)

