

Folder: DATA_EXTRACTED_WITH_99_FEATURES
------------------------------------------------------------
This folder contains the cleaned and preprocessed data extracted from the **UCI HAR Dataset** using raw inertial signal files.

✔️ SOURCE:
-----------
The raw data was taken from the Inertial Signals directory of the original UCI HAR dataset. These signals include 9 sensor signals:
- body_acc_x
- body_acc_y
- body_acc_z
- body_gyro_x
- body_gyro_y
- body_gyro_z
- total_acc_x
- total_acc_y
- total_acc_z

Each signal was recorded over 128 time steps for every activity window/sample.

✔️ PIPELINE:
------------
1. **Raw Data Loaded**  
   Inertial signal files from `train` and `test` directories were loaded and concatenated to form the full raw dataset.

2. **Raw Files Saved**  
   Raw signal data was saved as `X_RAW.csv` and labels as `Y_RAW.csv` for reference.

3. **Feature Extraction**  
   From each window (i.e., one sample of 128 time steps), the following statistical and frequency-domain features were extracted:
   
   For each signal (total 9), the following 11 features were computed:
   - Mean
   - Standard Deviation
   - Minimum
   - Maximum
   - Median
   - Skewness
   - Kurtosis
   - Energy (Sum of squares of FFT)
   - Entropy (of normalized FFT)
   - Mean Power (mean of FFT magnitudes)
   - Max Frequency Index (argmax of FFT)

   👉 This results in **99 features total** (9 signals × 11 features each).

4. **Cleaned Data Saved**  
   The extracted features were saved as:
   - `X_CLEAN.csv` — Cleaned feature matrix (shape: [samples, 99])
   - `Y_RAW.csv` — Corresponding labels for each sample
   - `feature_names.csv` — Names of the 99 features
   - `feature_names.txt` — Plain text list of all features
   - `shape_info.csv` — Shape of the final feature and label datasets

✔️ OUTPUT FILES:
----------------
- `X_CLEAN.csv` — Extracted feature dataset
- `Y_RAW.csv` — Activity labels
- `feature_names.csv` — Feature names in CSV
- `feature_names.txt` — Feature names in plain text
- `shape_info.csv` — Dataset shapes

📌 This dataset is now ready for use in machine learning pipelines for Human Activity Recognition (HAR).

Author: Priyam Pandey  
Date: [24 TH MAY 2025]








