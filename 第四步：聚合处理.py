import pandas as pd
import numpy as np

#如果直接在这个季度级别数据上做异常值分析，会出现以下问题：
#1. 同一个公司可能在某季度波动大，但整体是正常的
#比如某个公司 2016 年有个负利润特别极端，但整体 5 年平均利润是正常的，这时你就不该把它当异常。
#2. 同一个公司会被多次检查，造成重复异常判断
#比如同一家公司有 20 个季度记录，它 3 个季度超出了箱线图下限，那你就会统计“3 个异常”，但其实是一个公司行为。

import pandas as pd

# 读取两张 CSV 表
df1 = pd.read_csv('ST_2019_Corporate_Data_0327_11Variables.csv')
df2 = pd.read_csv('非ST_2019_Corporate_Data_0327_11Variables.csv')

# 上下拼接（行合并）
result = pd.concat([df1, df2], ignore_index=True)

# 保存结果
result.to_csv('merged_2019_IOM209_Data_0331.csv', index=False)

#进行聚合处理

ch=pd.read_csv("merged_2019_IOM209_Data_0331.csv")

ch.rename(columns={"ST or Not": "ST_flag"}, inplace=True)

# 设置要聚合的特征列（排除非数值列）
features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']

# 保留标签列 ST_flag
agg_ch = ch.groupby('code')[features + ['ST_flag']].mean().reset_index()

# 查看聚合后的结果
print(agg_ch.head())

agg_ch.to_csv('merged_aftergroup_0331.csv', index=False)

