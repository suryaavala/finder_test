import pandas as pd
import numpy as np


def load_data(file_path: str = "data/dataset_TakeHome.csv") -> pd.DataFrame:
    """Loads data from a csv file.

    Args:
        file_path (str): The name of the file to load. Should either be absolute or relative to the base repo.

    Returns:
        pd.DataFrame: dataset as dataframe
    """
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the dataframe by removing unnecessary columns and rows.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    df = _clean_data(df)
    df = _feature_selection(df)
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe by removing unnecessary columns and rows.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    df = df.drop_duplicates(inplace=True)
    df = df.dropna(inplace=True)
    return df


def _feature_selection(df: pd.DataFrame, corr_threshold: float = 0.7) -> pd.DataFrame:
    """Selects the features to use in the model.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    )
    cols_to_drop = [
        column
        for column in upper_tri.columns
        if any(upper_tri[column] > corr_threshold)
    ]
    print(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    return df
