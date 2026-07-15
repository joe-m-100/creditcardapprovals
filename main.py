from data_extractor import DataExtractor
from model import create_model

def run():
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

    print('Creating Model...')
    create_model(encoded_data)




# Main Program
run()