import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

# 1. 加载数据（含滞后变量的）
df = pd.read_csv("Individual Ready to Model.csv")

# 2. 特征列定义
base_features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
                 'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
                 'FA/TA', 'SE/FA', 'NP_SR']
lag1_features = [f"{col}_lag1" for col in base_features]
lag2_features = [f"{col}_lag2" for col in base_features]
all_features = base_features + lag1_features + lag2_features

X = df[all_features]
y = df['ST or Not']

# 3. 拆分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 定义 Pipeline + SMOTE + SVM
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('svm', SVC(probability=True, kernel='linear', random_state=42))
])

# 5. 搜索参数（如需非线性核可改 param_grid）
param_grid = {
    'svm__C': [0.1, 1, 10],
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    random_state=42
)

# 6. 模型训练
search.fit(X_train, y_train)

# 7. 模型预测与评估
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("✅ SVM 模型评估结果（加入滞后变量）")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 8. SHAP 可解释性分析（基于训练集解释测试集）
explainer = shap.Explainer(best_model.named_steps['svm'], X_train, feature_names=all_features)
shap_values = explainer(X_test)

# 8.1 Summary Plot
plt.figure()
shap.summary_plot(shap_values, features=X_test, feature_names=all_features, show=False)
plt.title("SHAP Summary Plot (SVM)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300)
plt.show()  # ← 显示图像

# 8.2 Bar Plot
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance (Bar) - SVM", fontsize=14)
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=300)
plt.show()  # ← 显示图像