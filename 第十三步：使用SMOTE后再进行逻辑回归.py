import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# 1. 读取标准化数据
df = pd.read_csv('final_model_data_quarterly_scaled.csv')

# 2. 特征与标签分离
features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
            'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
            'FA/TA', 'SE/FA', 'NP_SR']
X = df[features]
y = df['ST or Not']

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. SMOTE仅应用在训练集
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"SMOTE后训练集大小：{X_train_resampled.shape}, ST样本数：{sum(y_train_resampled)}")

# 5. 建立逻辑回归模型并训练
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# 6. 预测与评估
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("SMOTE逻辑回归模型评估结果")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (After SMOTE)')
plt.tight_layout()
plt.show()

# 8. 特征重要性可视化
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("特征重要性（SMOTE后）：")
print(coef_df)

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x='Coefficient',
    y='Feature',
    hue='Feature',
    data=coef_df,
    palette='coolwarm',
    dodge=False,
    legend=False
)
plt.title('Logistic Regression Feature Importance (After SMOTE)')
plt.axvline(0, color='gray', linestyle='--')

# 添加系数标签
for bar in ax.patches:
    width = bar.get_width()
    plt.text(
        width + 0.02 if width > 0 else width - 0.02,
        bar.get_y() + bar.get_height() / 2,
        f'{width:.2f}',
        ha='left' if width > 0 else 'right',
        va='center'
    )

plt.tight_layout()
plt.show()