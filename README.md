# Credit Card Fraud Detection

A machine learning project for detecting credit card fraud using XGBoost classifier with feature engineering and comprehensive evaluation metrics.

## Project Overview

This project implements a fraud detection system that:
- Performs feature engineering on transaction time data
- Trains an XGBoost classifier with class imbalance handling
- Evaluates model performance using multiple metrics (F1-score, Precision, Recall, ROC-AUC, PR-AUC)
- Visualizes results with ROC and Precision-Recall curves
## Model Performance
Train:
    F1: 0.9984 | Precision: 0.9967 | Recall: 1.0000
Val:
    F1: 0.8743 | Precision: 0.9481 | Recall: 0.8111
Test:
    F1: 0.8808 | Precision: 0.8854 | Recall: 0.8763

## Project Structure

```
creditCardFraudDetection/
├── code/
│   ├── credit_fraud_utils_data.py      # Data loading and feature engineering
│   ├── credit_fraud_utils_train.py     # Model training utilities
│   └── credit_fraud_utils_eval.py      # Model evaluation and visualization
├── data/
│   ├── train.csv                        # Training dataset
│   ├── val.csv                          # Validation dataset
│   ├── test.csv                         # Test dataset
│   └── trainval.csv                     # Combined train+val dataset
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

## Features

### Feature Engineering
- **Hour**: Extracted from transaction time (0-23)
- **Hour_sin**: Sinusoidal encoding of hour for cyclical patterns
- **Hour_cos**: Cosine encoding of hour for cyclical patterns
- **Day**: Day of week (0-6, where 0 is Monday)
- **Is_weekend**: Binary flag indicating weekend transactions

### Model
- **Algorithm**: XGBoost Classifier
- **Class Imbalance Handling**: Uses `scale_pos_weight` parameter
- **Hyperparameters**:
  - `n_estimators`: 200
  - `max_depth`: 10
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `eval_metric`: 'aucpr'

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd creditCardFraudDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Loading and Feature Engineering

```python
from code.credit_fraud_utils_data import load_data, feature_engineering

# Load data
train_df = load_data('data/train.csv')
val_df = load_data('data/val.csv')
test_df = load_data('data/test.csv')

# Feature engineering
x_train, y_train = feature_engineering(train_df)
x_val, y_val = feature_engineering(val_df)
x_test, y_test = feature_engineering(test_df)
```

### Model Training

```python
from code.credit_fraud_utils_train import Model

# Train model
model = Model(x_train, y_train)
```

### Model Evaluation

```python
from code.credit_fraud_utils_eval import evaluation

# Evaluate model
evaluation(model, x_train, y_train, x_val, y_val, x_test, y_test)
```

### Running Complete Pipeline

To run the complete pipeline (training + evaluation):

```bash
python code/credit_fraud_utils_eval.py
```

Or run individual components:

```bash
# Data processing
python code/credit_fraud_utils_data.py

# Model training
python code/credit_fraud_utils_train.py

# Model evaluation
python code/credit_fraud_utils_eval.py
```

## Evaluation Metrics

The evaluation function provides:
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve
- **Confusion Matrix**: Detailed breakdown of predictions

Results are displayed for:
- Training set
- Validation set
- Test set

## Data Requirements

The input CSV files should contain:
- **Time**: Transaction timestamp (in seconds)
- **V1-V28**: Anonymized features (PCA components)
- **Amount**: Transaction amount
- **Class**: Target variable (0 = normal, 1 = fraud)

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key packages include:
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib

## Notes

- The model uses `scale_pos_weight` to handle class imbalance automatically
- Feature engineering focuses on temporal patterns (hour, day, weekend)
- The evaluation includes both ROC and Precision-Recall curves, which are important for imbalanced datasets


## Author

[Maher Mohamed /Mail: maherosman800@gmail.com]
