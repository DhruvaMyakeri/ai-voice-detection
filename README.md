# 🎙️ AI Voice Deepfake Detection

### MFCC + Support Vector Machine (SVM)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

This project implements a high-precision machine learning pipeline to distinguish between authentic human speech and AI-generated synthetic voices. By leveraging **Mel-Frequency Cepstral Coefficients (MFCC)** and a **Support Vector Machine (SVM)**, the system identifies subtle spectral inconsistencies that are characteristic of "Deepfake" audio.

---

## 📌 Overview

The system classifies audio clips into two categories:

1.  **REAL:** Natural human speech.
2.  **FAKE:** AI-generated (deepfake) speech.

### ⚙️ Tech Stack

- **Feature Extraction:** `librosa` (MFCC)
- **Preprocessing:** `scikit-learn` (StandardScaler)
- **Classifier:** `scikit-learn` (SVM)
- **Visualization:** `matplotlib` & `seaborn`

---

## 🧠 Why This Works

AI-generated speech leaves "digital fingerprints" that differ from biological speech. While modern generative models are highly convincing, they frequently exhibit:

- **Over-smoothing:** Loss of high-frequency textures found in natural breath.
- **Structural Harmonics:** Synthetically structured harmonics that lack natural micro-variations.
- **Reduced Temporal Variation:** Minimal spectral flux over time compared to human speech.

MFCCs capture these spectral characteristics numerically, providing a "fingerprint" that the SVM uses to draw a clear decision boundary.

---

## 🔬 Methodology

### 1️⃣ Feature Extraction

For every audio file in the dataset:

- **Resampling:** All audio is standardized to **16kHz**.
- **MFCC Calculation:** 13 coefficients are computed to represent the power spectrum.
- **Aggregation:** Calculated the **Mean** and **Standard Deviation** for each coefficient to handle varying audio lengths.
- **Vectorization:** Each audio file results in a **26-dimension** feature vector:
  $$13 \text{ means} + 13 \text{ stds} = 26 \text{ features per audio}$$

### 2️⃣ Model Training

- **Scaling:** `StandardScaler` ensures all features contribute equally to the distance-based SVM.
- **Classification:** SVM with a Radial Basis Function (RBF) kernel and **class balancing**.
- **Splits:** Standard Training, Validation, and Testing sets.

---

## 📊 Results

The model demonstrates exceptional performance on the **Fake-or-Real (FoR)** dataset.

### Validation Metrics

| Class        | Precision | Recall | F1-Score | Support   |
| :----------- | :-------- | :----- | :------- | :-------- |
| **FAKE**     | 0.99      | 0.99   | 0.99     | 5398      |
| **REAL**     | 0.99      | 0.99   | 0.99     | 5400      |
| **Accuracy** |           |        | **0.99** | **10798** |

### Confusion Matrix

```text
                  Predicted FAKE    Predicted REAL
Actual FAKE            5356              42
Actual REAL              74              5326



Project Structure
Bash

ai-voice-detection/
├── notebooks/
│   ├── mfcc_exploration.ipynb   # EDA and signal analysis
│   └── pipelineAudio.ipynb      # Main training & evaluation pipeline
├── requirements.txt             # Project dependencies
├── .gitignore
└── README.md

Installation & Usage
Clone the repository:

Bash

git clone [https://github.com/your-username/ai-voice-detection.git](https://github.com/your-username/ai-voice-detection.git)
cd ai-voice-detection
Install dependencies:

Bash

pip install -r requirements.txt
Run the analysis:
Open notebooks/pipelineAudio.ipynb in Jupyter or VS Code to step through the extraction and training process.

🗂 Dataset
This project utilizes the Fake-or-Real (FoR) dataset (Normalized version).

Note: The dataset is not included in this repository due to its large size.

Plaintext

for-norm/
├── training/
├── validation/
└── testing/
```
