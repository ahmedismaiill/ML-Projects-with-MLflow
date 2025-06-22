🏦 Loan Approval Prediction Dataset

📄 Overview
This repository provides a synthetic dataset designed for classification tasks related to loan approval prediction. Inspired by the [original Kaggle dataset on Loan Approval Classification](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data), this version is enhanced with additional financial risk variables and scaled up using SMOTENC to improve class balance and size for machine learning experimentation.

📁 Dataset Metadata
Records: 45,000 samples
Features: 14 columns
Target Variable: loan_status (binary classification)

| Column Name                      | Description                                                    | Type        |
| -------------------------------- | -------------------------------------------------------------- | ----------- |
| `person_age`                     | Age of the person                                              | Float       |
| `person_gender`                  | Gender of the person                                           | Categorical |
| `person_education`               | Highest education level                                        | Categorical |
| `person_income`                  | Annual income in USD                                           | Float       |
| `person_emp_exp`                 | Years of employment experience                                 | Integer     |
| `person_home_ownership`          | Home ownership status (e.g., rent, own, mortgage)              | Categorical |
| `loan_amnt`                      | Loan amount requested                                          | Float       |
| `loan_intent`                    | Purpose of the loan (e.g., personal, education, medical, etc.) | Categorical |
| `loan_int_rate`                  | Interest rate on the loan                                      | Float       |
| `loan_percent_income`            | Loan amount as a percentage of the person’s income             | Float       |
| `cb_person_cred_hist_length`     | Length of credit history in years                              | Float       |
| `credit_score`                   | Credit score of the person                                     | Integer     |
| `previous_loan_defaults_on_file` | Indicator of previous loan defaults                            | Categorical |
| `loan_status` (Target)           | Loan approval status (1 = approved, 0 = rejected)              | Integer     |

📌 Usage

This dataset can be used for:
Binary classification modeling
Feature engineering and selection techniques
Model evaluation using metrics like accuracy, precision, recall, AUC
Testing imbalanced classification strategies (e.g., SMOTE, ADASYN)
Financial risk modeling
