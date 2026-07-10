import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def create_model(dataset: pd.DataFrame):

    # Create Train/Test datasets
    test_size = 0.2
    X_train, X_test, y_train, y_test = create_train_test_datasets(dataset, test_size)

    # Create Random Forest Classifier
    rfclassifier = RandomForestClassifier()

    rfclassifier.fit(X_train, y_train)

    y_pred = rfclassifier.predict(X_test) 

    # Display the first few predictions 
    print("First 5 predictions:", y_pred[:5]) 

    print("First 5 actual values:", y_test[:5]) 


def create_train_test_datasets(dataset: pd.DataFrame, test_size: float):
    X = dataset.drop(columns=['Approved'])
    y = dataset['Approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=15)

    return X_train, X_test, y_train, y_test