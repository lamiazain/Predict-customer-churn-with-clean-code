# Customer Churn Prediction Project

This project implements a machine learning pipeline to analyze customer churn data, perform feature engineering, train models, and evaluate their performance. It also includes a logging and testing framework to ensure reliability and reproducibility of results.

## 📌 Project Description
The goal of this project is to:
- Import and preprocess customer churn data.
- Perform Exploratory Data Analysis (EDA) and save plots.
- Engineer features for model training.
- Train two classification models:
    - Logistic Regression
    - Random Forest Classifier
- Save the trained models for future use.
- Generate:
    - ROC curves
    - Classification reports
    - Feature importance plots
- Provide unit tests with logging to validate each function in the pipeline.

## 📂 Files Description
1. `churn_library.py`
This is the main library file that contains:

- Data import and preprocessing functions.
- EDA functions (saving visualizations).
- Feature engineering pipeline.
- Model training, evaluation, and saving.
- Prediction functions.
- Report and visualization generators.
Key outputs include:
- ROC curves (`./data/roc_curves.png`)
- Classification reports (`./data/classification_report_rf.png`, `./data/classification_report_lr.png`)
- Feature importance plot (`./data/feture_importance_plot.png`)
- Saved models (`./models/rfc_model.pkl`, `./models/logistic_model.pkl`)

2. `churn_script_logging_and_tests.py`
This script:
- Contains unit tests for every function in `churn_library.py`.
- Logs success and error messages into `./logs/churn_library.log`.
- Validates that:
    - Files are created (plots, reports, models).
    - Data is split correctly.
    - Models generate predictions.
    - Plots and logs are non-empty.

## 📊 Data
The project uses:

- `./data/bank_data.csv` – Input dataset.
- Output artifacts (plots, reports, models) are saved in:

    - `./data/` – for EDA plots, ROC curves, reports, feature importance.
    - `./models/` – for trained models.
    - `./logs/` – for test logs.

## ⚙️ Requirements
- Python 3.7+
- Libraries:

    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - joblib
    - pytest
    - pylint
    - autopep8

## ▶️ Running the Code
Run the main library:
```python
python churn_library.py
```
Run the tests with logging:
```python
python churn_script_logging_and_tests.py
```
## 📝 Logging
All logs are saved to: `./logs/churn_library.log`

## ✅ Code Style Compliance
Both files are checked for PEP8 compliance using:
```python
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
```

## 📌 Summary
This project provides:
- A robust ML pipeline for churn prediction.
- Automated testing with logging.
- Easy reproducibility and extendibility for future improvements.