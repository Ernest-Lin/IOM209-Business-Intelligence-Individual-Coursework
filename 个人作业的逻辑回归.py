import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_csv("Individual Ready to Model.csv")

# 2. 特征定义
base_features = ['TA(t)/TA(t-1)', 'NP(t)/NP(t-1)', 'TL/TA', 'CA/CL',
                 'TL/TSE', 'NP/ASE', 'SR/ACA', 'MBC/AI',
                 'FA/TA', 'SE/FA', 'NP_SR']
lag1_features = [f"{col}_lag1" for col in base_features]
lag2_features = [f"{col}_lag2" for col in base_features]
all_features = base_features + lag1_features + lag2_features

X = df[all_features]
y = df['ST or Not']

# 3. 划分训练测试集（保持类别分布）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 拟合逻辑回归模型
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# 5. 模型评估
y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]

print("✅ 逻辑回归模型评估结果")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 6. SHAP 可解释性分析
explainer = shap.Explainer(lr_model, X_train, feature_names=all_features)
shap_values = explainer(X_test)

# 6.1 SHAP Summary Plot（综合重要性 + 密度）
plt.figure()
shap.summary_plot(shap_values, features=X_test, feature_names=all_features, show=False)
plt.title("SHAP Summary Plot (Logistic Regression)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300)
plt.show()

# 6.2 SHAP Bar Plot（平均绝对贡献排序）
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance (Bar)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=300)
plt.show()

# 6.3 可选：SHAP dependence plot（查看某个具体滞后变量的影响趋势）
# shap.dependence_plot("NP/ASE_lag1", shap_values.values, X_test, feature_names=all_features)