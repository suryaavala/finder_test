import pandas as pd
import numpy as np
from fraud_detector.utils import get_abs_path
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(file_path: str = "data/dataset_TakeHome.csv") -> pd.DataFrame:
    """Loads data from a csv file.

    Args:
        file_path (str): The name of the file to load. Should either be absolute or relative to the base repo.

    Returns:
        pd.DataFrame: dataset as dataframe
    """
    return pd.read_csv(get_abs_path(file_path))


def preprocess_data(
    df: pd.DataFrame, target_var: str = "Outcome"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses the dataframe by removing unnecessary columns and rows.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        Tuple[np.ndarray]: X_train, X_test, y_train, y_tes
    """
    df = _clean_data(df)
    df = _feature_selection(df)
    # X_train, X_test, y_train, y_test
    return train_test_split(
        df.drop(columns=[target_var]), df[target_var], test_size=0.2, random_state=42
    )


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe by removing unnecessary columns and rows.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
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
