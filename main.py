from data_extractor import DataExtractor

# Extract and clean the data from the CSV
de = DataExtractor('data/clean_dataset.csv')

raw_data = de.extract_data()
clean_data = de.clean_data(raw_data)

print('Data Extracted & Cleaned')