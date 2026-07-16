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

### Inital Results
These are the results from the first iteration of the model. There has been no parameter tuning performed yet, these results are to solely act as a baseline going forward.

* **Accuracy:** 86.232%
* **Precision:** 81.481%
* **Recall:** 83.019%

### Iteration Two Results - Manual Hyperparameter Tuning
These are the results from the second iteration of the model. In an attempt to reduce model overfitting, a maximum depth per tree has been introduced alongside a minimum number of leaf nodes (tentatively set to 2 - default is 1). This has led to a marginal increase in model accuracy, and a tradeoff between precision and recall of roughly 4%. As precision is being prioritised for this model, this is seen as an improvement over the initial model.

* **Accuracy:** 86.957%
* **Precision:** 85.714%
* **Recall:** 79.245%

*By increasing the number of trees from 200 to 500, accuracy increased by ~1% and precision increased by ~2%.*

*Model V2.1 results displayed below:*

* **Accuracy:** 87.681%
* **Precision:** 87.500%
* **Recall:** 79.245%

### Iteration Three Results - Hyperparameter Tuning using Cross Validation
For the third iteration of the model, a more stuctured approach was taken to hyperparameter tuning. To begin the iteration, `RandomizedSearchCV` was used to identify the most important hyperparameters. A search was ran with 200 iterations, and a wide spread of values for hyperparameters `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, and `max_features`. It was found that shallower trees of `max_depth` were preferable. 

As the running time for the search was not particularly long (< 5 mins), a more exhaustive search was performed using `GridSearchCV`. This concluded the following about the hyperparameters tested:

* `n_estimators` - Held very little value as no real trends were found across its values.
* `max_depth` - Shallower trees strongly correlate with better model performance. Tree depth greater than 10 reduces model accuracy significantly.
* `min_samples_leaf` - Smaller values (<= 5) are heavily preferred by the top models. High values for this parameter also harms model performance.
* `min_samples_split` - Very dependant on values of other hyperparameters. The best models used higher values for this hyperparameter (e.g. 20).
* `max_features` - Using all features correlates with poor classifier performance. `log2` or `sqrt` greatly preferred.
