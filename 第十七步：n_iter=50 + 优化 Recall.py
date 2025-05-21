from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline, Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取原始未标准化数据
df = pd.read_csv('Smote后数据.csv')

features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']
X = df[features]
y = df['ST or Not']

# 拆分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 构建 pipeline
pipeline = Pipeline([
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# 搜索空间
param_grid = {
    'xgb__n_estimators': [100, 150, 200, 250],
    'xgb__max_depth': [4, 5, 6, 7, 8],
    'xgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.7, 0.8, 1.0]
}

# 自动调参（关注 recall）
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=50,
    scoring='recall',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# 开始训练
search.fit(X_train, y_train)

# 评估最优模型
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("最佳参数组合：")
print(search.best_params_)

print("提升 Recall 后模型评估：")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Maximized Recall, XGBoost+SMOTE)')
plt.tight_layout()
plt.show()

# 9. 获取 xgb 模型
xgb_model = best_model.named_steps['xgb']

# 10. 特征重要性可视化
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 打印重要性数值
print("XGBoost 特征重要性（按贡献排序）:")
print(importance_df)

# 绘图展示
plt.figure(figsize=(8, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()