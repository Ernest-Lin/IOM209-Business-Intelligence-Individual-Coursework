import pandas as pd
import numpy as np

agg_ch = pd.read_csv('merged_aftergroup_0331.csv')

# 1. 设定财务特征列
features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']

# 2. 筛选非 ST 公司
non_st = agg_ch[agg_ch['ST_flag'] == 0]

# 3. IQR 异常值检测函数
def detect_outliers_iqr(df, features):
    outlier_summary = {}
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = {
            '异常值个数': len(outliers),
            '占比': round(len(outliers) / len(df) * 100, 2),
            '下界': round(lower_bound, 3),
            '上界': round(upper_bound, 3)
        }
    return pd.DataFrame(outlier_summary).T.sort_values(by='异常值个数', ascending=False)

# 4. 执行异常值检测
outlier_report = detect_outliers_iqr(non_st, features)
print(outlier_report)

#删除异常值公司（基于部分关键特征）
#只删除异常值过多的变量对应的公司（非 ST）
# 定义要剔除异常值的关键变量
remove_outlier_cols = ['NP(t)/NP(t-1)', 'NP/ASE', 'FA/TA', 'SR/ACA']

# 拿出非 ST 样本
non_st = agg_ch[agg_ch['ST_flag'] == 0]

# 初始化需要保留的行索引集合
mask = pd.Series(True, index=non_st.index)

# 对每个变量执行 IQR 逻辑剔除异常公司
for col in remove_outlier_cols:
    Q1 = non_st[col].quantile(0.25)
    Q3 = non_st[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask &= (non_st[col] >= lower) & (non_st[col] <= upper)

# 得到处理后的非 ST 公司
cleaned_non_st = non_st[mask]

# 合并 ST 公司 + 干净的非 ST 公司
st = agg_ch[agg_ch['ST_flag'] == 1]
agg_ch_cleaned = pd.concat([st, cleaned_non_st], ignore_index=True)

# 查看新样本数量
print(f"原始非 ST 公司数：{len(non_st)}")
print(f"清洗后非 ST 公司数：{len(cleaned_non_st)}")
print(f"总样本数（含 ST）：{len(agg_ch_cleaned)}")

agg_ch_cleaned.to_csv('merged_aftergroup_cleaned_0331.csv', index=False)

