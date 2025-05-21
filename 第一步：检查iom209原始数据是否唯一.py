import pandas as pd
import numpy as np

ST_2019_Corporate_Data=pd.read_csv("ST_2019_Corporate_Data.csv")

ST_2019_Corporate_Data["EndDate"] = pd.to_datetime(ST_2019_Corporate_Data["EndDate"])
ST_2019_Corporate_Data = ST_2019_Corporate_Data.sort_values(by=["code", "EndDate"])
ST_2019_Corporate_Data

ST_2019_Corporate_Data.duplicated(subset = ['code','EndDate']).unique()

ch=pd.read_csv("所有2019_Corporate_Data.csv")

ch["EndDate"] = pd.to_datetime(ch["EndDate"])

ch = ch.sort_values(by=["code", "EndDate"])

ch.duplicated(subset = ['code','EndDate']).unique()