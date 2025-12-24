# Stress vs Unstressed Classification using PCA and SVM  
*A Dimensionality-Reduced Machine Learning Pipeline for Physiological Emotion Analysis*

---

## Overview
This repository presents a **classical machine learning pipeline** for binary emotion classification (**Stressed vs Unstressed**) using **Principal Component Analysis (PCA)** for dimensionality reduction followed by a **Support Vector Machine (SVM)** classifier with an RBF kernel.

The project is designed as a **methodologically clean and interpretable baseline**, suitable for inclusion in a **PhD research portfolio**, particularly in contexts involving:
- Affective computing  
- Physiological signal analysis  
- Stress detection systems  
- Explainable and classical ML pipelines  

---

## Research Motivation
High-dimensional physiological datasets often suffer from:
- Redundant features  
- Noise amplification  
- Increased computational cost  

This work investigates:
- Whether **variance-preserving dimensionality reduction (PCA)** can improve model stability
- How **non-linear SVM decision boundaries** perform on reduced feature spaces
- The trade-off between **model complexity and interpretability**

The emphasis is on **robust preprocessing and principled modeling**, not deep learning.

---

## Dataset Description
- Source: `dataset.csv`
- Target Variable: `Emotion`
- Classes:
  - `Stressed`
  - `neutral`, `relaxed` → merged into `Unstressed`
- Metadata columns (`timestamps`, `Subject`) are removed

### Class Balancing Strategy
To avoid class imbalance:
- 50,000 samples selected from the *Stressed* class
- 25,000 samples selected from the beginning and end of the *Unstressed* class
- Final dataset is balanced before training

---

## Preprocessing Pipeline

### 1. Label Encoding
- `Stressed → 1`
- `Unstressed → 0`

### 2. Feature Scaling
- **StandardScaler** applied to normalize all features
- Ensures compatibility with distance-based and margin-based models

### 3. Dimensionality Reduction (PCA)
- PCA retains **95% of total variance**
- Automatically determines optimal number of components
- Reduces redundancy and improves numerical stability

---

## Model Architecture

### Support Vector Machine (SVM)
- Kernel: Radial Basis Function (RBF)
- Hyperparameters:
  - `C = 100`
  - `gamma = 0.1`
- Probabilistic outputs enabled (`probability=True`)

SVM is selected for its:
- Strong performance in high-dimensional spaces
- Ability to model non-linear decision boundaries
- Solid theoretical grounding

---

## Training and Evaluation

### Data Split
- 70% Training
- 30% Testing
- Fixed random seed for reproducibility

### Evaluation Metrics
- Overall Accuracy
- Precision, Recall, F1-score (via classification report)
- Confusion Matrix visualization
- PCA explained variance analysis

---

## Results and Visualization

### Confusion Matrix
- Visualized using a Seaborn heatmap
- Highlights class-wise prediction performance

### PCA Explained Variance Plot
- Shows how variance is distributed across retained components
- Demonstrates effective dimensionality compression with minimal information loss

---

## Key Contributions
- Clean end-to-end ML pipeline
- Explicit class balancing strategy
- Variance-preserving dimensionality reduction
- Interpretable classical model
- Strong baseline for comparison with deep learning approaches

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  

---

## Research Significance
This project demonstrates:
- Strong understanding of **classical machine learning fundamentals**
- Ability to design **statistically principled pipelines**
- Awareness of **bias–variance and dimensionality trade-offs**

It serves as a **solid baseline** for:
- EEG-based stress detection
- Multimodal affective systems
- Hybrid ML–DL comparative studies

---

## Future Extensions
- Hyperparameter optimization via grid or Bayesian search
- Comparison with linear SVM and logistic regression
- Feature importance analysis in PCA space
- Integration with deep feature extractors
- Subject-wise cross-validation for physiological generalization

---

## Reproducibility
All preprocessing steps, sampling strategies, and model parameters are explicitly defined to ensure **full reproducibility** and experimental transparency.

---
