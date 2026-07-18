import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def create_model(dataset: pd.DataFrame) -> None:

    # Create Train/Test datasets
    test_size = 0.2
    X_train, X_test, y_train, y_test = create_train_test_datasets(dataset, test_size)

    # Create Random Forest Classifier
    rfclassifier = RandomForestClassifier(
        n_estimators=300,
        random_state=15,
        min_samples_split=25,
        min_samples_leaf=2,
        max_depth=4,
        max_features='sqrt',
    )

    rfclassifier.fit(X_train, y_train)
    predictions = rfclassifier.predict(X_test)

    score_model(predictions, y_test)


def create_train_test_datasets(dataset: pd.DataFrame, test_size: float) -> tuple:
    X = dataset.drop(columns=['Approved'])
    y = dataset['Approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=15)

    return X_train, X_test, y_train, y_test

def score_model(y_pred, y_true) -> None:
    acc = accuracy_score(y_true, y_pred)
    prec = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))

    print(f'\nModel Evaluation:\nAccuracy: {round((acc*100), 3)}\nPrecision: {round((prec*100), 3)}\nRecall: {round((recall*100), 3)}\n')

def search(dataset: pd.DataFrame, type: str, param_distributions: dict, k_folds: int | None, iterations: int = 10) -> None:

    X_train, X_test, y_train, y_test = create_train_test_datasets(dataset=dataset, test_size=0.2)

    if type == 'randomized':
        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(), 
            param_distributions=param_distributions, 
            n_iter=iterations,
            cv=k_folds,
            random_state=15,
            n_jobs=-1
        )
    elif type == 'grid':
        search = GridSearchCV(
            estimator=RandomForestClassifier(), 
            param_grid=param_distributions, 
            cv=k_folds,
            n_jobs=-1
        )
    else:
        return None

    search.fit(X_train, y_train)

    results_df = pd.DataFrame(search.cv_results_)

    # Display Results
    top_results = results_df.sort_values(by='mean_test_score', ascending=False).head(10)
    top_results.reset_index()

    bottom_results = results_df.sort_values(by='mean_test_score', ascending=False).tail(10)
    bottom_results.reset_index()

    for index, row in top_results.iterrows():
        print(row['rank_test_score'])
        print(f'n_estimators: {row['param_n_estimators']} min_samples_split: {row['param_min_samples_split']} max_depth: {row['param_max_depth']} min_samples_leaf: {row['param_min_samples_leaf']} max_features: {row['param_max_features']}')
 

    for index, row in bottom_results.iterrows():
        print(row['rank_test_score'])
        print(f'n_estimators: {row['param_n_estimators']} min_samples_split: {row['param_min_samples_split']} max_depth: {row['param_max_depth']} min_samples_leaf: {row['param_min_samples_leaf']} max_features: {row['param_max_features']}')
