from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 载入数据
X_scaled_df = pd.read_csv('X_scaled_0331.csv')

# 1. 提取标准化特征
features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']

X = X_scaled_df[features]

# 2. 执行 PCA（降到 2 维）
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
#
# # 3. 构建包含主成分的 DataFrame
# pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
# pca_df['code'] = X_scaled_df['code'].values
# pca_df['ST_flag'] = X_scaled_df['ST_flag'].values
#
# # 4. 分离 ST 和 非 ST 的数据
# pca_st = pca_df[pca_df['ST_flag'] == 1]
# pca_nonst = pca_df[pca_df['ST_flag'] == 0]
#
# # 图 1：只画 ST 公司（红色）
# plt.figure(figsize=(6, 5))
# sns.scatterplot(data=pca_st, x='PC1', y='PC2', color='red', s=60)
# plt.title('PCA - ST Companies Only (Red)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # 图 2：只画非 ST 公司（蓝色）
# plt.figure(figsize=(6, 5))
# sns.scatterplot(data=pca_nonst, x='PC1', y='PC2', color='blue', s=60)
# plt.title('PCA - Non-ST Companies Only (Blue)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # 图 3：合图（红点覆盖蓝点）
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=pca_nonst, x='PC1', y='PC2', color='blue', s=60, label='Non-ST (0)')
# sns.scatterplot(data=pca_st, x='PC1', y='PC2', color='red', s=60, label='ST (1)')
# plt.title('PCA - ST vs Non-ST (Red ST Covers Blue Non-ST)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.legend(title='Company Type')
# plt.tight_layout()
# plt.show()




from sklearn.neighbors import NearestNeighbors

# 用标准化后的原始特征空间进行匹配
# 取出 ST 和 非 ST 的标准化特征部分
X_st = X_scaled_df[X_scaled_df['ST_flag'] == 1]
X_nonst = X_scaled_df[X_scaled_df['ST_flag'] == 0]

# 只提取标准化的财务特征
X_st_feat = X_st[features].values
X_nonst_feat = X_nonst[features].values

# 建立 KNN 模型（使用欧氏距离）
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_nonst_feat)

# 对每个 ST 公司找到其最近的 3 个非 ST 公司
distances, indices = knn.kneighbors(X_st_feat)

# 创建匹配结果表格
matched_codes = []

for i, neighbor_idxs in enumerate(indices):
    st_code = X_st.iloc[i]['code']
    for idx in neighbor_idxs:
        matched_code = X_nonst.iloc[idx]['code']
        matched_codes.append({
            'ST_code': st_code,
            'Matched_nonST_code': matched_code
        })

matched_df = pd.DataFrame(matched_codes)

# 查看前几条匹配结果
print(f"共找到匹配关系数：{len(matched_df)}")
print(f"唯一非 ST 匹配公司数：{matched_df['Matched_nonST_code'].nunique()}")

matched_df.to_csv('KNN_Code.csv', index=False)