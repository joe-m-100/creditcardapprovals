import unittest
from data_extractor import DataExtractor

class TestDataExtractor(unittest.TestCase):

    def test_data_extracted_and_cleaned(self):
        # Given
        de = DataExtractor('data/test_data.csv')

        # When
        raw_data = de.extract_data()
        clean_data = de.clean_data(raw_data)

        expected_length = 8

        # Then
        self.assertEqual(expected_length, len(clean_data))

