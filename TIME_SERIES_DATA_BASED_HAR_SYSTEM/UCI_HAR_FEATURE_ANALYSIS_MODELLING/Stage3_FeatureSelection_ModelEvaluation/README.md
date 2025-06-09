# Stage 3: Feature Selection and Enhanced Modeling

This stage enhances performance and interpretability via **feature selection** on both 99-feature and 561-feature datasets.

---

## ✅ Feature Selection Techniques Used:
- ANOVA F-value
- Chi-Squared Test
- Recursive Feature Elimination (RFE)
- Forward Feature Selection
- Feature Importance (Random Forest, Decision Tree)

### 🔍 Optimal Feature Count:
- Selected using **KneeLocator** from the `kneed` library.

---

## 🧪 Model Training:
- Models trained on top 20, 40, 60 selected features.
- Evaluation of same 12 models per subset.

---

## 📦 Output Folder: `OUTPUT_PPD`
- Knee plots, score plots, top-k feature plots
- Model performance tables for each feature selection method
- Ranked feature importance charts
