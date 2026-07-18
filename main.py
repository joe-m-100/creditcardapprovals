import time
import pandas as pd
from data_extractor import DataExtractor
from model import create_model, search

def setup() -> pd.DataFrame:
    # Extract and clean the data from the CSV
    de = DataExtractor('data/clean_dataset.csv')

    raw_data = de.extract_data()
    clean_data = de.clean_data(raw_data)

    print('Data Extracted & Cleaned.')
    print('Raw data rows: ', len(raw_data))
    print('Clean data rows: ', len(clean_data))

    pos, neg = de.get_data_balance(clean_data, 'Approved')
    print(f'\nData Balance - {pos} positve cases, {neg} negative cases\n')


    # One-Hot encode all fields that are not numerical
    encoded_data = de.hot_one_encode(clean_data, ['Industry', 'Ethnicity', 'Citizen']) 
    print('One-Hot Encoded non-numerical fields.')

    return encoded_data


def param_search(dataset: pd.DataFrame, search_type: str):
    distributions = {
        'n_estimators': [200, 300, 400],
        'max_depth': [4],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [25, 30, 35],
        'max_features': ['sqrt']
    }

    if search_type == 'randomized':
        search(
            dataset=dataset,
            type='randomized',
            param_distributions=distributions,
            iterations=200,
            k_folds=10
        )
    elif search_type == 'grid':
        search(
            dataset=dataset,
            type='grid',
            param_distributions=distributions,
            k_folds=10
        )

def run(dataset: pd.DataFrame):
    print('Creating Model...')
    create_model(dataset)


if __name__ == '__main__':
    start_time = time.time()

    data = setup()

    # param_search(data, 'grid')
    run(data)

    # Display runtime
    time_elapsed = time.time() - start_time
    print(f'Runtime: {time_elapsed // 60} mins {round(time_elapsed % 60, 2)} secs')