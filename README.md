# IOM209-Business-Intelligence-Individual-Coursework
# From Signals to Distress: A Time-Aware Machine Learning Framework for ST Classification in China’s Capital Market

## Overview

This project presents a time-aware machine learning framework designed to predict whether listed companies in China's A-share market are likely to receive the "Special Treatment" (ST) classification, which serves as a regulatory indicator of financial distress. By combining multi-period financial indicators, classical machine learning models, and SHAP-based interpretability, this study captures the temporal dynamics underlying corporate deterioration.

## Highlights

- **Dataset**: Firm-level quarterly financial data from 2015Q1 to 2019Q4.
- **Target**: 51 ST-labeled firms in 2019 vs. matched non-ST counterparts.
- **Features**: 11 core financial ratios with $t$, $t-1$, and $t-2$ lags (33 total features).
- **Models Used**: Logistic Regression, SVM, Random Forest, XGBoost.
- **Techniques**:
  - Data preprocessing (standardization, outlier removal, IQR filtering).
  - Sample balancing using **KNN-matching** and **SMOTE**.
  - SHAP for model interpretation and temporal importance analysis.

> ⚠️ **Note**: Due to the large size of processed `.csv` and intermediate `.txt` files (e.g., full lag-augmented datasets, raw CSMAR exports, and SMOTE results), these files are excluded from the GitHub repository.

If you are interested in accessing the full dataset or reproducing the complete pipeline, feel free to contact me via:

- 📧 `3050139891@qq.com`
- 📧 `ernestlin2004@163.com`

## Citation

If you find this project helpful, feel free to cite it or acknowledge the work.


## Contact

Developed by **Ernest Lin**, School of Information Management and Information Systems, Xi’an Jiaotong-Liverpool University.
