from data_extractor import DataExtractor

def run():
    # Extract and clean the data from the CSV
    de = DataExtractor('data/clean_dataset.csv')

    raw_data = de.extract_data()
    clean_data = de.clean_data(raw_data)

    print('Data Extracted & Cleaned')
    print('Raw data rows: ', len(raw_data))
    print('Clean data rows: ', len(clean_data))



# Main Program
run()