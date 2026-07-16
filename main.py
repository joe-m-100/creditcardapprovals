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
        'n_estimators': [200, 500, 800],
        'max_depth': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'min_samples_split': [2, 5, 10, 20],
        'max_features': ['sqrt', 'log2', None]
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
            iterations=200,
            k_folds=10
        )

def run(dataset: pd.DataFrame):
    print('Creating Model...')
    create_model(dataset)


if __name__ == '__main__':
    data = setup()

    # Main Program
    param_search(data, 'grid')