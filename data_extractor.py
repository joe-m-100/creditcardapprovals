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
    
    def hot_one_encode(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        encoded_df = pd.get_dummies(df, columns=columns, dtype=int)

        return encoded_df
