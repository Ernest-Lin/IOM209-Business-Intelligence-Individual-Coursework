import pandas as pd
import numpy as np

ch=pd.read_csv("所有2019_Corporate_Data.csv")

ch["EndDate"] = pd.to_datetime(ch["EndDate"])

ch = ch.sort_values(by=["code", "EndDate"])

#偿债能力

ch["TL/TA"] = ch["负债合计"] / ch["资产总计"]
ch["CA/CL"] = ch["流动资产合计"] / ch["流动负债合计"]
ch["(CA-I)/CL"] = (ch["流动资产合计"] - ch["存货净额"]) / ch["流动负债合计"]
ch["TL/TSE"] = ch["负债合计"] / ch["所有者权益合计"]
ch["CL/TA"] = ch["流动负债合计"] / ch["资产总计"]
ch["EBIT/IE"] = ch["息税前利润（EBIT）"] / ch["利息支出"]
ch.rename(columns={"经营活动产生的现金流量净额／流动负债": "NOCF/CL"}, inplace=True)

#盈利能力

def calculate_rolling_average(group, col_name):
    shifted = group[col_name].shift(1)
    avg = (group[col_name] + shifted) / 2
    avg_filled = avg.fillna(group[col_name])
    return avg_filled

columns_to_average = {
    "资产总计": "平均总资产",
    "流动资产合计": "平均流动资产",
    "固定资产净额": "平均固定资产",
    "所有者权益合计": "平均股东权益",
    "存货净额": "平均存货"
}

for original_col, new_col in columns_to_average.items():
    if original_col in ch.columns:
        ch[new_col] = ch.groupby("code", group_keys=False).apply(
            lambda g: calculate_rolling_average(g, original_col)
        )
    else:
        print(f"字段 {original_col} 不存在，跳过计算 {new_col}")


ch["EBIT/ATA"] = ch["息税前利润（EBIT）"] / ch["平均总资产"]
ch["NP/ATA"] = ch["净利润"] / ch["平均总资产"]
ch["NP/ACA"] = ch["净利润"] / ch["平均流动资产"]
ch["NP/AFA"] = ch["净利润"] / ch["平均固定资产"]
ch["NP/ASE"] = ch["净利润"] / ch["平均股东权益"]

#营运能力

ch.rename(columns={"主营业务利润占比": "MBI/ATA"}, inplace=True)

ch.rename(columns={"总营业成本率TTM": "TTM"}, inplace=True)

ch["SR/ACA"] = ch["营业收入"] / ch["平均流动资产"]
ch["SR/AFA"] = ch["营业收入"] / ch["平均固定资产"]

ch["MBC/AI"] = ch["营业成本"] / ch["平均存货"]

#发展能力

ch.rename(columns={"营业总收入增长率": "MBI(t)/MBI(t-1)"}, inplace=True)

ch.rename(columns={"总资产增长率A": "TA(t)/TA(t-1)"}, inplace=True)

ch.rename(columns={"净利润增长率A": "NP(t)/NP(t-1)"}, inplace=True)

#结构稳健性

ch["CA/TA"] = ch["流动资产合计"] / ch["资产总计"]
ch["FA/TA"] = ch["固定资产净额"] / ch["资产总计"]
ch["SE/FA"] = ch["所有者权益合计"] / ch["资产总计"]
ch["CL/TL"] = ch["流动负债合计"] / ch["负债合计"]

#资本扩张能力

ch.rename(columns={"每股净资产0": "TA/NOS"}, inplace=True)
ch.rename(columns={"每股收益0": "NP/NOS"}, inplace=True)

ch["NP_SR"] = ch["净利润"] / ch["营业收入"]

#剔除st的code

# 读取要剔除的 code 列表
with open("ST_2019_Codetemplate.txt", "r") as f:
    codes_to_remove = [line.strip() for line in f.readlines() if line.strip()]

# 将其转换为字符串（确保和 ch["code"] 格式一致）
codes_to_remove = [code.zfill(6) for code in codes_to_remove]  # 保证前导0齐全

# 如果 ch["code"] 是 int 类型，先转为字符串再比对
ch["code"] = ch["code"].astype(str).str.zfill(6)

# 剔除这些行
ch = ch[~ch["code"].isin(codes_to_remove)]

ch.to_csv("非ST_2019_Corporate_Data_0323111.csv",index=False)

ch.duplicated(subset = ['code','EndDate']).unique()

ch