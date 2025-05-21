import pandas as pd
import numpy as np

ch=pd.read_csv("ST_2019_Corporate_Data_0327.csv")

ch

columns_to_keep = [
    'code', 'EndDate',
    'TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
    'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
    'FA/TA', 'SE/FA', 'NP_SR', 'ST or Not'
]

ch = ch.loc[:, columns_to_keep]

ch = ch.sort_values(by=["code", "EndDate"])

import numpy as np

# 替换 inf/-inf 为 NaN
ch.replace([np.inf, -np.inf], np.nan, inplace=True)

# 你要填充的变量列
features_to_impute = [
    'TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
    'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
    'FA/TA', 'SE/FA', 'NP_SR'
]

# 用每一列的均值进行填充
for col in features_to_impute:
    ch[col].fillna(ch[col].mean(), inplace=True)

ch

ch.to_csv("ST_2019_Corporate_Data_0327_11Variables.csv",index=False)

import pandas as pd
import numpy as np

ch=pd.read_csv("非ST_2019_Corporate_Data_0327.csv")

ch

columns_to_keep = [
    'code', 'EndDate',
    'TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
    'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
    'FA/TA', 'SE/FA', 'NP_SR', 'ST or Not'
]

ch = ch.loc[:, columns_to_keep]

ch = ch.sort_values(by=["code", "EndDate"])

import numpy as np

# 替换 inf/-inf 为 NaN
ch.replace([np.inf, -np.inf], np.nan, inplace=True)

# 你要填充的变量列
features_to_impute = [
    'TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
    'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
    'FA/TA', 'SE/FA', 'NP_SR'
]

# 用每一列的均值进行填充
for col in features_to_impute:
    ch[col].fillna(ch[col].mean(), inplace=True)

ch

ch.to_csv("非ST_2019_Corporate_Data_0327_11Variables.csv",index=False)