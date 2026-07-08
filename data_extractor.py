import pandas as pd

class DataExtractor:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def extract_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filename)

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        no_empty_cells = df.dropna()
        no_duplicates = no_empty_cells.drop_duplicates()

        return no_duplicates