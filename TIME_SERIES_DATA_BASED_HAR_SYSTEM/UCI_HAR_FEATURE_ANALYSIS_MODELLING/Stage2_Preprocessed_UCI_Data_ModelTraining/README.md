# 📦 Stage 2: Modeling with UCI Preprocessed 561-Feature Dataset

This stage uses the **already preprocessed** dataset provided by the UCI HAR dataset, containing **561 time and frequency domain features** per sample.

---

## 🗂️ Dataset Used

- `X_train.txt`, `y_train.txt`
- `X_test.txt`, `y_test.txt`

These contain 561 precomputed features such as:
- Time-domain: mean, std, correlation, energy
- Frequency-domain: FFT coefficients, bands energy

---

## 🤖 Models Trained

Same 12 classifiers as in Stage 1.

---

## 📈 Outputs in `OUTPUT_PPD/`

- Accuracy, F1-score, precision, recall for each model.
- Confusion matrices.
- Visual comparisons with Stage 1 performance.

---

## 🎯 Goal

To establish a benchmark using engineered features from domain experts and compare against custom feature engineering from Stage 1.
