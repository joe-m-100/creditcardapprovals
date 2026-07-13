# Credit Card Approval Predictor

A machine learning project built with Python and `scikit-learn` to predict whether a credit card application will be approved or denied based on applicant details and financial history.

## Project Overview
Evaluating credit risk is a binary classification problem. This project automates the approval workflow by processing a mix of categorical and numerical applicant data to predict the final approval decision (`Approved` vs. `Denied`).


## Dataset
The dataset used for this project is **not** included in this repository.

* **Source:** https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data
* **Size:** 690 rows, 16 features
* **Target Variable:** `Approved` (Binary: 0 = Denied, 1 = Approved)

## Models
This project uses a random forest classifier as its model. This was chosen due to the relatively small dataset used (< 1000 rows) to avoid overfitting. 

Due to the nature of the data being used, the aim of this project was to create a model with a high precision score (% of correct positive predictions). This is because falsely admitting a high risk individual could have costly financial ramifications for the adminstering bank/firm, so minimiing false positives is a priority.

### Setup and Usage:
1. Clone the repository.
2. Download the `clean_dataset.csv` from the source link above.
3. Ensure the filepath to the dataset passed into the `DataExtractor` object in `main.py` is correct.
4. Navigate to the root `creditcardapprovals` directory.
5. Run `python main.py`.

## Results

# Inital Results
* **Accuracy:** 86.232%
* **Precision:** 81.481%
* **Recall:** 83.019%