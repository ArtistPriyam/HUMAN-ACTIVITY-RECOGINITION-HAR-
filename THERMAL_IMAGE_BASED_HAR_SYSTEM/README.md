
## Thermal Image-Based Human Activity Recognition (HAR)

### Project Title:

**ThermalPoseHAR: Human Activity Recognition using Thermal Images and Pose Estimation**

---

## 1. Objective

The goal of this project is to develop and deploy a deep learning-based pipeline that can recognize human activities from **thermal images** using **pose estimation** (via YOLOv8-Pose) and a **classifier model** (e.g., MLP). This solution is intended for surveillance, safety, and behavior monitoring in low-visibility conditions.

---

## 2. importance 

* Traditional RGB-based activity recognition fails in low-light or obscured scenarios.
* Thermal cameras are increasingly used in security, defense, and industrial environments.
* Combining pose estimation with activity classification offers a lightweight, interpretable, and privacy-aware solution.

---

## 3. Pipeline Overview

```
Input Thermal Image (.png/.jpg)
        ↓
YOLOv8-Pose Model (custom-trained on thermal images)
        ↓
Keypoint Vector Extraction (17 keypoints → 51-dim feature vector)
        ↓
Classifier Model (e.g., MLP, SVM, LSTM)
        ↓
Predicted Activity Label (e.g., running, sitting, jumping)
```

---

## 4. ⚙️ Technologies Used

| Component        | Framework/Library            |
| ---------------- | ---------------------------- |
| Pose Estimation  | YOLOv8-Pose (Ultralytics)    |
| Feature Handling | NumPy, OpenCV                |
| Classifier       | Scikit-learn (MLPClassifier) |
| Web App          | Flask                        |
| Deployment       | Flask Web Server             |

---

## 5.  Model Details

### 🔸 YOLOv8-Pose

* Pretrained on MS COCO keypoints.
* Fine-tuned on thermal imagery dataset.
* Outputs 17 keypoints per detected person.

### 🔸 Feature Vector

* (x, y) coordinates + confidence for 17 joints = 51 features.
* Normalized w\.r.t image dimensions.

### 🔸 MLP Classifier

* Input: 51-dim feature vector.
* Hidden Layers: \[128, 64]
* Activation: ReLU
* Output: Softmax over 10 activity classes.
---

## 7. 🖼️ Sample Activities

| Index | Activity Label        |
| ----- | --------------------- |
| 0     | walking               |
| 1     | LOOKING\_STRAIGHT     |
| 2     | STANDING              |
| 3     | jumping\_climbing     |
| 4     | suspicious\_look      |
| 5     | EXERCISE\_BODY\_SWING |
| 6     | SITTING\_STANDING     |
| 7     | fighting              |
| 8     | gesturing             |
| 9     | LOOKING\_UP           |

---

## 8. 🌐 Flask Web Application

### Features:

* Upload a thermal image.
* Backend extracts keypoints using YOLOv8-Pose.
* Classifies the activity using MLP.
* Displays predicted activity label with image preview.

### How it Works:

```bash
pip install - e .
$ python app.py
# Opens http://127.0.0.1:5000
```
## 10. 📊 Future Improvements

* Deploy using Docker + Gunicorn or Render.
* Support real-time webcam or video stream inference.
* Replace MLP with LSTM for temporal modeling.
* Add multiple-person support.
* Optimize for edge deployment (e.g., Jetson Nano).

---

## 11. 👤 Authors

* **Name**: PRIYAM PANDEY

---

## 12. 📎 Appendix

### 🔸 Dependencies

```bash
pip install ultralytics scikit-learn opencv-python flask
```

## Project Directory Structure

```
THERMAL_IMAGE_BASED_HAR_SYSTEM/
├── .github/                          # GitHub configurations (workflows, actions)
├── artifacts/                        # Saved models, weights, and prediction outputs
│   ├── classifiers/                 # Trained classifier models (e.g., mlp.pkl)
│   ├── weights/                     # YOLOv8-Pose trained weights
│   └── TEST/                        # Test thermal images
├── config/                           # Configuration files (optional YAMLs, params)
├── logs/                             # Logging files during training or inference
├── research/                         # Jupyter notebooks for experimentation and results
│   ├── 01_data_ingestion.ipynb       # data ingestion
│   ├── DL_MODEL_TRAINING.ipynb       # lstm model training
│   ├── ML_CLASSIFIER_MODELS_TRAINING.ipynb # mlp classifer and random forest training 
│   ├── PREDICTION.ipynb 
│   ├── YOLOv8_model_training.ipynb   # training YOLOv8_model 
│   └── trials.ipynb                  # labelling tools code 
├── src/                              # Source code
│   └── predict.py                    # HAR_Predictor class for inference
├── static/                           # Static assets (images, CSS, etc.)
│   └── uploads/                     # Uploaded images for web app
├── templates/                        # HTML templates for Flask
│   └── index.html
├── app.py                            # Flask web app script
├── setup.py                          # Setup script for packaging (if needed)
├── requirements.txt                  # Project dependencies
├── README.md                         # Project overview and usage instructions
└── template.py                       # (Possibly unused or template script)
```

---

### Notes:

* `research/` contains all experimentation and training notebooks for reproducibility.
* `src/` handles modular code for predictions, making it production-friendly.
* `static/uploads/` and `templates/index.html` support the Flask frontend.
* `artifacts/` holds model artifacts created during training and testing.
* `app.py` is the entry point for the deployed web application.
----------------------------------------------------------------------------------------

# 🔖 Activity Labeling(CODE IN UTILS.ipynb file)

To train the classifier, each image is assigned an activity label. These labels are saved in `activity_labels.csv`, which is used during training. Labeling was performed using a custom tool provided in:

> 📄 `research/utils.ipynb`
> 🛠️ `labeling_tool.py`

---

### 🛠️ labeling\_tool.py – Quick Labeling Utility

**Purpose:**
Display thermal images one-by-one and assign activity labels via keyboard input.

**How to Run:**

```bash
python labeling_tool.py
```

**Key Controls:**

| Keys   | Action                |
| ------ | --------------------- |
| 1–9, 0 | Assign activity label |
| k      | Skip image            |
| Esc    | Exit tool             |

**Output Files:**

* `activity_labels.csv` – Class index to label mapping
* `activity_labels_train.csv` – Image path and label pairs for training
**Directory to Label:**
Ensure this path is correctly set:

```python
DATA_DIR = "artifacts/OPEN_THERMAL_IMAGE/val/images"
``