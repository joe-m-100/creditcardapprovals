# Credit Card Approval Predictor

A machine learning project built with Python and `scikit-learn` to predict whether a credit card application will be approved or denied based on applicant details and financial history.

## Project Overview
Evaluating credit risk is a binary classification problem. This project automates the approval workflow by processing a mix of categorical and numerical applicant data to predict the final approval decision (`Approved` vs. `Denied`).


## 📊 Dataset
To comply with data privacy and repository size best practices, the dataset is **not** included in this repository. 

* **Source:** https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data
* **Size:** 690 rows, 16 features
* **Target Variable:** `Approved` (Binary: 0 = Denied, 1 = Approved)

### Setup and Usage:
1. Download the `clean_dataset.csv` from the source link above.
2. Ensure the filepath passed into the ```DataExtractor``` object in ```main.py``` is correct.
3. Navigate to the root ```creditcardapprovals``` directory.
3. Run ```python main.py```.