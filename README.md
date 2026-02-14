# Machine Learning Classification Models - Assignment 2

**Student ID:** 2025AB05021  
**Assignment:** ML Classification Models Comparison  
**Date:** 15 February 2026

## a. Problem Statement

This project develops a comprehensive machine learning classification framework that can be applied to diverse binary and multi-class classification problems. While demonstrated using the Breast Cancer Wisconsin dataset, the system is designed to be dataset-agnostic and can handle any classification task with appropriate features.

The primary objective is to build, evaluate, and compare six different machine learning classification algorithms on user-provided datasets through an interactive web application. This enables users to upload their own datasets and instantly compare model performance across multiple metrics, facilitating informed model selection for their specific classification problems.

The system aims to democratize machine learning by providing an accessible, no-code interface for comparing classification algorithms, making it valuable for researchers, data scientists, and domain experts who need to quickly evaluate which models work best for their particular classification challenges.


## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source:** UCI Machine Learning Repository
**Type:** Binary Classification
**Objective:** Predict whether a breast tumor is malignant (M) or benign (B) based on features computed from digitized images of fine needle aspirate (FNA) of breast masses.

### Dataset Characteristics:
- **Number of instances:** 569
- **Number of features:** 30 
- **Target Variable** Diagnosis (2 classes)
M = Malignant (Cancer)
B = Benign (Non-cancerous)
- **Missing Values** None

**Class Distribution**
**Benign** 212 instances (37.3%)
**Malignant** 357 instances (62.7%)

**Feature Information**
The 30 features represent characteristics of cell nuclei present in the image. For each cell nucleus, 10 real-valued features are computed, and for each feature, three values are provided:


### Feature Categories:
The 30 features are computed from cell nuclei characteristics and include:

1. **Mean values (10 features):** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
2. **Standard error values (10 features):** SE for each of the above mean values
3. **"Worst" values (10 features):** Mean of the three worst (largest) values for each feature.

The dataset is well-balanced and contains high-quality features that are highly relevant for cancer diagnosis prediction.

## c. Models Used

### Performance Comparison Table
================================================================================
                 Model  Accuracy  AUC Score  Precision  Recall  F1 Score    MCC
   Logistic Regression    0.9825     0.9954     0.9861  0.9861    0.9861 0.9623
         Decision Tree    0.9123     0.9157     0.9559  0.9028    0.9286 0.8174
   K-Nearest Neighbors    0.9561     0.9788     0.9589  0.9722    0.9655 0.9054
Naive Bayes (Gaussian)    0.9298     0.9868     0.9444  0.9444    0.9444 0.8492
         Random Forest    0.9561     0.9937     0.9589  0.9722    0.9655 0.9054
               XGBoost    0.9649     0.9917     0.9474  1.0000    0.9730 0.9258

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-----------------------------------|
| Logistic Regression | **Best Overall Performer**: Achieves the top accuracy (98.25%), AUC (99.54%) and MCC (0.9623), showing excellent class separation and balanced precision/recall. The model’s strong linear performance and low variance suggest the dataset is largely linearly separable; logistic regression is fast, interpretable (feature coefficients), and is a reliable baseline for production when explainability and stability matter. It may underperform if there are complex non-linear feature interactions. |
| Decision Tree : Lower performance, Accuracy (91.23%) and AUC (91.57%) are the lowest among models; precision (95.59%) exceeds recall (90.28%), indicating conservative positive predictions and some missed positives. Decision trees are highly interpretable and useful for feature-importance insights, but they are prone to overfitting and instability on small data changes; pruning or using as part of an ensemble generally improves generalization. |
| k-Nearest Neighbors : Strong neighborhood signal, High accuracy (95.61%), AUC (97.88%) and very high recall (97.22%) indicate that similar instances share labels and that local structure is informative. KNN benefits from proper scaling and choice of k; it is simple and non-parametric but can be slow at inference for large datasets and sensitive to irrelevant features. |
| Naive Bayes (Gaussian) : Well-calibrated probabilities, Good accuracy (92.98%) and very high AUC (98.68%) show strong ranking ability and calibrated probability outputs despite the conditional-independence assumption. Precision and recall are balanced (94.44%), so it is a reliable, lightweight option when fast training/inference and probability estimates are needed; performance may degrade when features are strongly correlated.
Random Forest — Robust ensemble, Matches KNN’s accuracy (95.61%) but with even higher AUC (99.37%), indicating excellent ranking and probability estimation. The forest reduces variance and handles non-linear interactions and missing signals well; it’s a strong general-purpose choice with built-in feature importance, though it is heavier computationally and less interpretable than single trees.|
| XGBoost (Ensemble) : High-recall ensemble: Second-best accuracy (96.49%) with perfect recall (100%) and strong F1/MCC, meaning it successfully detects all positive cases in this split but yields slightly more false positives (precision 94.74%). XGBoost effectively models complex interactions and is highly tunable — ideal when recall (sensitivity) is critical — but requires careful hyperparameter tuning and regularization to avoid overfitting and to balance precision/recall.|


## Files Structure:
```
├── README.md                           # This documentation
├── streamlit_app.py                   # Main Streamlit application
├── requirements.txt                   # Python dependencies
├── model_results.csv                  # Training evaluation results
├── metadata.pkl                       # Dataset and training metadata
├── scaler.pkl                        # Feature scaling transformer
├── *_model.pkl                       # Trained model files (6 models)
└── ML_classification_Models.ipynb    # Training notebook
```

## Usage:
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run streamlit_app.py`
3. Upload CSV data or use the default breast cancer dataset
4. Select a model and view real-time evaluation metrics
