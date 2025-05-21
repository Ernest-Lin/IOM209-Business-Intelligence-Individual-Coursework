from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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

# 构建pipeline（SMOTE + RandomForest）
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

# 随机森林专用参数搜索空间
param_grid = {
    'rf__n_estimators': [100, 150, 200, 250, 300],
    'rf__max_depth': [None, 10, 15, 20, 25],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2', 0.5, 0.7],
    'rf__bootstrap': [True, False]
}

# 随机搜索，其实就是自动调参（同样关注recall）
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

print("\n=== 随机森林最佳参数 ===")
print(search.best_params_)

print("\n=== 评估指标 ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 可视化混淆矩阵
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Random Forest+SMOTE)')
plt.tight_layout()
plt.show()

# 获取 rf 模型
rf_model = best_model.named_steps['rf']

# 特征重要性可视化
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n=== 特征重要性 ===")
print(importance_df)

# 特征重要性可视化（横向条形图）
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()