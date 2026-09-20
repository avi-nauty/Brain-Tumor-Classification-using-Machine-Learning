# 🧠 Brain Tumor Detection from MRI Images using Machine Learning

> A deep learning research project for binary classification of brain MRI scans (Tumor vs. Healthy) using fused CNN feature extraction, Black Hole Optimization (BHO) feature selection, and a custom Optimized SGD Logistic Regression classifier.

📄 **Paper submitted to Springer Nature — awaiting first round of review**

---

## 📌 Overview

This project presents an end-to-end machine learning pipeline for detecting brain tumors from MRI images. Instead of relying on a single model, it fuses deep features from three pre-trained CNN backbones into a rich 3,840-dimensional feature vector, applies Black Hole Optimization to select the most discriminative features, and trains a custom Optimized Stochastic Gradient Descent (Opt-SGD) logistic classifier from scratch.

The pipeline was developed as a Master's research project at Lakehead University and run on Kaggle using GPU-accelerated Jupyter Notebooks.

---

## 🏆 Results

| Metric | Score |
|---|---|
| **Accuracy** | **98.48%** |
| Precision | High |
| Sensitivity (Recall) | High |
| Specificity | High |
| F1-Score | High |

> Results evaluated on a held-out test set. The BHO-based feature selection significantly reduced computational cost while preserving classification performance.

---

## 🔬 Pipeline Architecture

```
MRI Images
    │
    ▼
┌─────────────────────────────────────────┐
│         Image Preprocessing             │
│  Letterbox Resize → 224×224 → [0,1]     │
│  + Light MRI-safe Augmentation          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│       Deep Feature Extraction           │
│                                         │
│  ResNet50   →  2,048 features           │
│  VGG16      →    512 features           │
│  EfficientNetB0 → 1,280 features        │
│                                         │
│  Fused → 3,840-dimensional vector       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│   Black Hole Optimization (BHO)         │
│   Feature Selection                     │
│   3,840 → ~2,000 optimal features       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│   Opt-SGD Logistic Regression           │
│   Custom implementation from scratch    │
│   Binary: Tumor vs. Healthy             │
└─────────────────────────────────────────┘
    │
    ▼
  Prediction + Evaluation Metrics
```

---


## 🗂️ Datasets
 
Three publicly available MRI datasets were used to evaluate the pipeline, referred to as Dataset 1, Dataset 2, and Dataset 3 in this repository.
 
| # | Name | Glioma Tumor Images | Healthy Brain Images | Source |
|---|---|---|---|---|
| Dataset 1 | BRISC 2025 | 1,401 | 1,207 | [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) |
| Dataset 2 | SciDB + Masoud Kaggle 2025 | 1,291 | 1,282 | [Kaggle](https://www.kaggle.com/dsv/11957028) |
| Dataset 3 | Mendeley 2025 | 3,773 | 2,432 | [Mendeley Data](https://data.mendeley.com/datasets/c9rt8d6zrf/1) |
 
### Dataset Details
 
**Dataset 1 — BRISC 2025**
A recently introduced benchmark dataset for MRI-based brain tumor classification. It provides high-quality labeled scans covering a broad range of tumor sizes, shapes, and locations, making it well-suited for testing model performance across varied clinical presentations.
 
**Dataset 2 — SciDB + Masoud Kaggle 2025**
A composite dataset formed by merging the SciDB medical imaging collection with Masoud Nickparvar's Brain Tumor MRI Dataset from Kaggle. Combining scans from multiple independent sources introduces greater data diversity, which helps models learn more generalizable features across different tumor morphologies.

 
**Dataset 3 — Mendeley 2025**
A publicly available MRI dataset curated specifically for brain tumor detection tasks. It spans multiple tumor categories captured under varying imaging conditions, offering a realistic and challenging testbed for evaluating how well a model holds up in clinical settings.
 
---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 📁 Project Structure

```
brain-tumor-detection/
│
├── bho-opt-sgd-classifier.ipynb   # Main Kaggle notebook (full pipeline)
│
├── data/                          # MRI dataset (train/test split)
│   ├── train/
│   │   ├── healthy/
│   │   └── tumor/
│   └── test/
│       ├── healthy/
│       └── tumor/
│
├── saved_features/                # Precomputed feature arrays
│   ├── X_train_fused.npy          # 3,840-dim fused features (train)
│   ├── X_test_fused.npy           # 3,840-dim fused features (test)
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── X_train_bho.npy            # BHO-selected features (train)
│   └── X_test_bho.npy             # BHO-selected features (test)
│
├── saved_models/
│   ├── opt_sgd_weights.npy        # Trained classifier weights
│   ├── opt_sgd_bias.npy
│   └── opt_sgd_results.npy        # Full evaluation metrics
│
└── README.md
```

---

## ⚙️ Key Components

### 1. Image Preprocessing
- Letterbox resize to 224×224 (aspect-preserving with zero-padding)
- Pixel normalization to [0, 1]
- MRI-safe light augmentation: random rotation (±10°), translation (5%), zoom (5%), contrast (5%)
- Backbone-specific `preprocess_input` applied per model

### 2. Feature Extraction (Transfer Learning)
Three ImageNet-pretrained CNN backbones used as frozen feature extractors with Global Average Pooling:

| Backbone | Output Features |
|---|---|
| ResNet50 | 2,048 |
| VGG16 | 512 |
| EfficientNetB0 | 1,280 |
| **Fused** | **3,840** |

All three feature vectors are concatenated to form a rich, multi-scale representation of each MRI image.

### 3. Black Hole Optimization (BHO) Feature Selection
A metaheuristic optimization algorithm inspired by black hole physics. BHO iteratively selects the most discriminative subset of features from the 3,840-dimensional fused vector, reducing it to approximately 2,000 optimal features while preserving classification accuracy and reducing inference time.

### 4. Custom Opt-SGD Logistic Classifier
A logistic regression classifier built entirely from scratch using a hybrid optimization strategy:
- **Full gradient anchor** computed at the start of each epoch
- **Stochastic inner updates** using a random number of mini-steps per epoch
- **L2 regularization** applied to weights only (not bias)
- Trained for 150 epochs with learning rate 0.005

---

## 🚀 How to Run

This notebook was developed and run on **Kaggle** with GPU acceleration.

### Run on Kaggle
1. Upload the notebook to [Kaggle](https://www.kaggle.com)
2. Enable GPU accelerator under **Settings → Accelerator → GPU**
3. Attach the MRI dataset
4. Run all cells in order

### Run Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

# Install dependencies
pip install tensorflow keras scikit-learn numpy pandas matplotlib

# Launch Jupyter
jupyter notebook bho-opt-sgd-classifier.ipynb
```

> **Note:** GPU is strongly recommended for feature extraction. CPU-only runs will be significantly slower.

---

## 📊 Dataset

- **Format:** MRI images (PNG/JPG/JPEG/BMP)
- **Classes:** `healthy` (0), `tumor` (1)
- **Split:** Train / Test directories
- **Preprocessing:** Balanced class distribution verified before training

---

## 📚 Research & Publication

This work was conducted as a Master of Science research project at **Lakehead University**, Thunder Bay, ON, Canada (2024–2026).

- **Supervisor:** Dr. Hosam El-Ocla, PhD, P. Eng.
Associate Professor, Department of Computer Science, Lakehead University, Thunder Bay, ON, Canada.
- **Paper:** Submitted to **Springer Nature** — currently awaiting first round of review

---

## 👤 Author

**Aviral Nautiyal**
M.S. Computer Science — Lakehead University (3.9 GPA)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/aviral-nautiyal)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/avinauty)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
