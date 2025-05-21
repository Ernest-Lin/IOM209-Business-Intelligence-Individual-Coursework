import pandas as pd
import numpy as np

#先找2019年所有a股公司，查看在2019年为st的公司，前五年的季报数据，去计算各种指数---1
#2015-2020年
#1:k-means knn
#2:最近邻算法-knn。找到50家st公司的每一家最近的3家非st公司，maybe小于150
#3:预测st是否为1，在此基础上做一个特征重要性


ch=pd.read_csv("查询ST公司.csv")

ch

# 第一步：确保 FirstSTDate 是 datetime 类型
ch['FirstSTDate'] = pd.to_datetime(ch['FirstSTDate'], errors='coerce')

# 第二步：筛选年份为 2019 的行
ch_2019 = ch[ch['FirstSTDate'].dt.year == 2019]

ch_2019.to_csv("ST_2019_Corporate.csv",index=False)

ch_2019_Symbol= ch_2019.loc[:, ['Symbol',]]

ch_2019_Symbol['Symbol'] = ch_2019_Symbol['Symbol'].astype(str).str.zfill(6)

ch_2019_Symbol.to_csv('Codetemplate.txt', index=False, header=False)

#计算ST公司的财务数据

ST_2019_Corporate_Data=pd.read_csv("ST_2019_Corporate_Data.csv")

ST_2019_Corporate_Data = ST_2019_Corporate_Data.sort_values(by='code')

ST_2019_Corporate_Data

#开始计算
#偿债能力

ST_2019_Corporate_Data["TL/TA"] = ST_2019_Corporate_Data["负债合计"] / ST_2019_Corporate_Data["资产总计"]
ST_2019_Corporate_Data["CA/CL"] = ST_2019_Corporate_Data["流动资产合计"] / ST_2019_Corporate_Data["流动负债合计"]
ST_2019_Corporate_Data["(CA-I)/CL"] = (ST_2019_Corporate_Data["流动资产合计"] - ST_2019_Corporate_Data["存货净额"]) / ST_2019_Corporate_Data["流动负债合计"]
ST_2019_Corporate_Data["TL/TSE"] = ST_2019_Corporate_Data["负债合计"] / ST_2019_Corporate_Data["所有者权益合计"]
ST_2019_Corporate_Data["CL/TA"] = ST_2019_Corporate_Data["流动负债合计"] / ST_2019_Corporate_Data["资产总计"]
ST_2019_Corporate_Data["EBIT/IE"] = ST_2019_Corporate_Data["息税前利润（EBIT）"] / ST_2019_Corporate_Data["利息支出"]
ST_2019_Corporate_Data.rename(columns={"经营活动产生的现金流量净额／流动负债": "NOCF/CL"}, inplace=True)

#盈利能力

ST_2019_Corporate_Data["EndDate"] = pd.to_datetime(ST_2019_Corporate_Data["EndDate"])

ST_2019_Corporate_Data = ST_2019_Corporate_Data.sort_values(by=["code", "EndDate"])

def calculate_rolling_average(group, col_name):
    shifted = group[col_name].shift(1)
    avg = (group[col_name] + shifted) / 2
    avg_filled = avg.fillna(group[col_name])
    return avg_filled

columns_to_average = {
    "资产总计": "平均总资产",
    "流动资产合计": "平均流动资产",
    "固定资产净额": "平均固定资产",
    "所有者权益合计": "平均股东权益"
}

for original_col, new_col in columns_to_average.items():
    if original_col in ST_2019_Corporate_Data.columns:
        ST_2019_Corporate_Data[new_col] = (
            ST_2019_Corporate_Data
            .groupby("code", group_keys=False)
            .apply(lambda g: calculate_rolling_average(g, original_col))
        )
    else:
        print(f"字段 {original_col} 不存在，跳过计算 {new_col}")


ST_2019_Corporate_Data["EBIT/ATA"] = ST_2019_Corporate_Data["息税前利润（EBIT）"] / ST_2019_Corporate_Data["平均总资产"]
ST_2019_Corporate_Data["NP/ATA"] = ST_2019_Corporate_Data["净利润"] / ST_2019_Corporate_Data["平均总资产"]
ST_2019_Corporate_Data["NP/ACA"] = ST_2019_Corporate_Data["净利润"] / ST_2019_Corporate_Data["平均流动资产"]
ST_2019_Corporate_Data["NP/AFA"] = ST_2019_Corporate_Data["净利润"] / ST_2019_Corporate_Data["平均固定资产"]
ST_2019_Corporate_Data["NP/ASE"] = ST_2019_Corporate_Data["净利润"] / ST_2019_Corporate_Data["平均股东权益"]

#营运能力

ST_2019_Corporate_Data.rename(columns={"主营业务利润占比": "MBI/ATA"}, inplace=True)

ST_2019_Corporate_Data.rename(columns={"总营业成本率TTM": "TTM"}, inplace=True)

ST_2019_Corporate_Data["SR/ACA"] = ST_2019_Corporate_Data["营业收入"] / ST_2019_Corporate_Data["平均流动资产"]
ST_2019_Corporate_Data["SR/AFA"] = ST_2019_Corporate_Data["营业收入"] / ST_2019_Corporate_Data["平均固定资产"]


if "存货净额" in ST_2019_Corporate_Data.columns:
    ST_2019_Corporate_Data["平均存货"] = ST_2019_Corporate_Data.groupby("code", group_keys=False).apply(
        lambda g: calculate_rolling_average(g, "存货净额")
    )

print("平均存货" in ST_2019_Corporate_Data.columns)

ST_2019_Corporate_Data["MBC/AI"] = ST_2019_Corporate_Data["营业成本"] / ST_2019_Corporate_Data["平均存货"]

#发展能力

ST_2019_Corporate_Data.rename(columns={"营业总收入增长率": "MBI(t)/MBI(t-1)"}, inplace=True)

ST_2019_Corporate_Data.rename(columns={"总资产增长率A": "TA(t)/TA(t-1)"}, inplace=True)

ST_2019_Corporate_Data.rename(columns={"净利润增长率A": "NP(t)/NP(t-1)"}, inplace=True)

#结构稳健性
ST_2019_Corporate_Data["CA/TA"] = ST_2019_Corporate_Data["流动资产合计"] / ST_2019_Corporate_Data["资产总计"]
ST_2019_Corporate_Data["FA/TA"] = ST_2019_Corporate_Data["固定资产净额"] / ST_2019_Corporate_Data["资产总计"]
ST_2019_Corporate_Data["SE/FA"] = ST_2019_Corporate_Data["所有者权益合计"] / ST_2019_Corporate_Data["资产总计"]
ST_2019_Corporate_Data["CL/TL"] = ST_2019_Corporate_Data["流动负债合计"] / ST_2019_Corporate_Data["负债合计"]

#资本扩张能力

ST_2019_Corporate_Data.rename(columns={"每股净资产0": "TA/NOS"}, inplace=True)
ST_2019_Corporate_Data.rename(columns={"每股收益0": "NP/NOS"}, inplace=True)

ST_2019_Corporate_Data["NP_SR"] = ST_2019_Corporate_Data["净利润"] / ST_2019_Corporate_Data["营业收入"]

ST_2019_Corporate_Data.to_csv("ST_2019_Corporate_Data_0323.csv",index=False)

ST_2019_Corporate_Data.duplicated(subset = ['code','EndDate']).unique()

ST_2019_Corporate_Data