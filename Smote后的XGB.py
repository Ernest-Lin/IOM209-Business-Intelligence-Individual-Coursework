from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（假设已处理过SMOTE的数据）
df = pd.read_csv('Ready to Model.csv')

features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']
X = df[features]
y = df['ST or Not']

# 拆分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 构建纯XGBoost流程
pipeline = Pipeline([
    ('xgb', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=1  # 假设数据已平衡
    ))
])

# 参数搜索空间（优化参数）
param_grid = {
    'xgb__n_estimators': [100, 150, 200, 250],
    'xgb__max_depth': [4, 5, 6, 7, 8],
    'xgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'xgb__gamma': [0, 0.1, 0.2],
    'xgb__reg_alpha': [0, 0.1, 1],
    'xgb__reg_lambda': [0, 0.1, 1]
}

# 优化配置（使用平衡评估指标）
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=50,
    scoring='balanced_accuracy',  # 平衡准确率
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# 训练模型
search.fit(X_train, y_train)

# 模型评估
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("最佳参数组合：")
print(search.best_params_)

print("\n=== 模型性能 ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵可视化
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('XGBoost混淆矩阵')
plt.tight_layout()
plt.show()

# 特征重要性分析
xgb_model = best_model.named_steps['xgb']
importances = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    '特征': features,
    '重要性': importances
}).sort_values(by='重要性', ascending=False)

print("\n=== 特征重要性排序 ===")
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(8, 6))
sns.barplot(data=importance_df, x='重要性', y='特征', palette='viridis')
plt.title('XGBoost特征重要性分析')
plt.tight_layout()
plt.show()