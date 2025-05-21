import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
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

# 3. 拆分训练集与测试集（保持分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 定义SMOTE + SVM的Pipeline
pipeline = Pipeline([
    ('svm', SVC(probability=True, random_state=42))  # 启用概率估计
])

# 5. 网格搜索参数
# param_grid = {
#     'svm__C': [0.1, 1, 10],  # 正则化参数
#     'svm__kernel': ['linear', 'rbf'],  # 核函数
#     'svm__gamma': ['scale', 'auto']  # RBF核的参数
# }

param_grid = {
    'svm__C': [0.1, 1, 10],  # 正则化参数
    'svm__kernel': ['linear'],  # 核函数

}


# 6. 网格搜索（按AUC评分优化）
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=20,  # 仅随机尝试20次（远小于GridSearch的穷举）
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

#开始训练
random_search.fit(X_train, y_train)



# 7. 获取最佳模型
best_model = random_search.best_estimator_
print(f"\n🎯 最佳参数组合: {random_search.best_params_}")

# 8. 预测与评估
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 9. 输出评估指标
print("\n✅ SVM模型评估结果（SVM）")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 10. 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - SVM with SMOTE')
plt.tight_layout()
plt.show()

# 11. 特征重要性（仅当使用线性核时）
if best_model.named_steps['svm'].kernel == 'linear':
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': best_model.named_steps['svm'].coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print("\n📌 特征重要性（线性核的权重系数，按绝对值排序）:")
    print(coef_df)

    # 可视化特征权重
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
    plt.title('SVM Feature Importance (Linear Kernel)')
    plt.axvline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ 注意：当前使用非线性核（{}），无法直接解释特征重要性".format(
        best_model.named_steps['svm'].kernel))