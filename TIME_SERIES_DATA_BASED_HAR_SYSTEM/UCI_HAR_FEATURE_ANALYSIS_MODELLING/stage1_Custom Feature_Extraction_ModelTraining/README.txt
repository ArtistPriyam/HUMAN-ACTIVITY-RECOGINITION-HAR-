# 🔧 Stage 1: Custom Feature Engineering and Machine Learning

In this stage, we begin from the **raw inertial signals** (body acceleration, gyroscope, jerk) and apply handcrafted statistical techniques to extract **99 custom features**.

---

## 🛠️ Data Processing

- Raw data merged and reshaped from separate files.
- Extracted 99 features using:
  - Statistical measures: mean, std, median, min, max, IQR
  - Domain-specific metrics: signal energy, magnitude, entropy, correlation
- Labels mapped to 6 human activities.

---

## 🤖 Machine Learning Models Trained

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- XGBoost
- AdaBoost
- Gaussian Naive Bayes
- Multi-layer Perceptron (MLP)
- Extra Trees Classifier
- Bagging Classifier

---

## 📈 Outputs in `OUTPUT_PPD/`

- Classification metrics (accuracy, precision, recall, F1).
- Confusion matrices.
- Comparative model performance plots.

---

## ✅ Goal

To test how well handcrafted statistical features from raw signals can classify human activity.
