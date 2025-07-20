import pandas as pd


def normalize_df(df:pd.DataFrame) -> pd.DataFrame:
    return (df-df.mean()) / df.std()