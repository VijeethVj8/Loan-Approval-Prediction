# Loan Approval Prediction

This project focuses on predicting loan approval status using a variety of machine learning models, including ensemble methods and hyperparameter tuning. The main goal is to build a highly accurate model to predict whether a loan will be approved based on demographic, financial, and credit history features.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)

## Project Overview
The Loan Approval Prediction project aims to build a predictive model that can determine whether a loan application will be approved or denied. We use various machine learning techniques, including ensemble models and stacked classifiers, to improve prediction accuracy.

## Dataset
The dataset contains several features related to loan applicants, including:
- **person_age**: The age of the applicant.
- **person_income**: The annual income of the applicant.
- **person_home_ownership**: The home ownership status of the applicant.
- **person_emp_length**: The number of years the applicant has been employed.
- **loan_intent**: The purpose for which the loan is applied.
- **loan_grade**: The loan risk grade.
- **loan_amnt**: The loan amount applied for.
- **loan_int_rate**: The interest rate of the loan.
- **cb_person_cred_hist_length**: The length of the applicant's credit history.

The target variable is `loan_status`, where:
- 1: Approved
- 0: Denied

## Feature Engineering
To enhance model performance, we engineered additional features:
- **income_to_loan_ratio**: Ratio of the person's income to the loan amount.
- **cred_hist_loan_amnt_ratio**: Ratio of credit history length to loan amount.
- **age_income_ratio**: Ratio of the applicant's age to their income.
- **age_cred_hist_interaction**: Interaction between age and credit history length.

## Modeling
We trained several models including:
- **XGBoost**
- **Random Forest**
- **LightGBM**
- **CatBoost**

These models were then stacked using **Logistic Regression** as the meta-model to combine their predictions and improve performance.

## Hyperparameter Tuning
We used **GridSearchCV** to perform hyperparameter tuning for the Logistic Regression meta-model. The tuned parameters include:
- **C**: Regularization strength
- **solver**: Optimization algorithm
- **penalty**: Type of regularization

## Results
The final **Stacked Model** achieved a **ROC-AUC score of 0.9584** on the validation set. The key evaluation metrics are:
- Precision: 96%
- Recall: 75%
- F1-score: 0.82

## How to Run the Project
### Requirements
- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `matplotlib`, `seaborn`

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/loan-approval-prediction.git
    cd loan-approval-prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook to see the full analysis and results:
    ```bash
    jupyter notebook Loan.ipynb
    ```

## Conclusion
This project demonstrated how to effectively use feature engineering, ensemble models, and hyperparameter tuning to improve loan approval prediction accuracy. The final model achieved strong results, with potential for further improvement by experimenting with additional features or advanced stacking techniques.
