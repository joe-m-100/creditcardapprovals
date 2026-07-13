import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def create_model(dataset: pd.DataFrame):

    # Create Train/Test datasets
    test_size = 0.2
    X_train, X_test, y_train, y_test = create_train_test_datasets(dataset, test_size)

    # Create Random Forest Classifier
    rfclassifier = RandomForestClassifier(n_estimators=185, random_state=15, ccp_alpha=0.0)
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

    