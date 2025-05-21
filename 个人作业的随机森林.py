import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 1. 加载数据（包含滞后变量）
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

# 4. Pipeline：SMOTE + RF
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

# 5. 随机搜索参数
param_grid = {
    'rf__n_estimators': [100, 150, 200],
    'rf__max_depth': [4, 6, 8, None],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    random_state=42
)

# 6. 训练
search.fit(X_train, y_train)
best_model = search.best_estimator_

# 7. 评估
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("✅ Random Forest 模型评估结果（加入滞后变量）")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 8. SHAP 可解释性分析
explainer = shap.TreeExplainer(best_model.named_steps['rf'])
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):  # 二分类时取正类
    shap_values = shap_values[1]

# 8.1 SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=all_features, show=False)
plt.title("SHAP Summary Plot (Random Forest)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary_plot_rf.png", dpi=300)
plt.show()

