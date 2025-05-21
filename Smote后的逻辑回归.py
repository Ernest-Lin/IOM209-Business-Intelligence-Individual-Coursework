import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载标准化数据
df = pd.read_csv('Ready to Model.csv')

# 2. 特征与标签
features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']
X = df[features]
y = df['ST or Not']

# 3. 拆分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. 建立模型并训练
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. 预测与评估
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. 输出评估指标
print("✅ 逻辑回归模型评估结果")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 7. 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 8. 输出变量权重（特征重要性）
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n📌 特征重要性（按绝对系数排序）:")
print(coef_df)

# 9. 可视化特征权重
plt.figure(figsize=(8, 5))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
plt.title('Logistic Regression Feature Importance')
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()