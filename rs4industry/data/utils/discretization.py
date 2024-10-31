import time
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


def timestamp_to_weekday_hour_min(df, col: str):
    """ Process the timestamp column to weekday, hour and minute
    Args:
        df (pd.DataFrame): dataframe with a timestamp column
        col (str): name of the timestamp column
    Returns:
        pd.DataFrame: dataframe with the new columns
    """
    # Corrected the time handler function
    def _time_handler(timestamp):
        ts_struct = time.localtime(timestamp // 1000)
        ts_wday = ts_struct.tm_wday + 1
        ts_hour = ts_struct.tm_hour + 1
        ts_min = ts_struct.tm_min + 1
        return ts_wday, ts_hour, ts_min
    
    # apply the function to the timestamp column, get three columns
    df[col] = df[col].astype('int64')
    df[[col + '_wday', col + '_hour', col + '_min']] = df[col].apply(lambda x: pd.Series(_time_handler(x)))
    # delete the original timestamp column
    df = df.drop(columns=[col])
    return df


def standard_scaler(df, col):
    """ Standardize the column of df using the StandardScaler
    Args:
        df (pd.DataFrame): dataframe to be scaled
        col (str): name of the column to be scaled
    Returns:
        pd.DataFrame: dataframe with the new column
    """
    scaler = StandardScaler()
    scaler.fit(df[[col]])
    df[col] = scaler.transform(df[[col]])
    return df


def kbins_discretizer(df, col, n_bins=5):
    """ Discretize the columns of df using the KBinsDiscretizer
    Args:
        df (pd.DataFrame): dataframe to be discretized
        col (str): name of the column to be discretized
        n_bins (int): number of bins
    Returns:
        pd.DataFrame: dataframe with the new column
    """
    kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    kbins.fit(df[[col]])
    df[col] = kbins.transform(df[[col]])
    return df

DISCRITIZER_MAP = {
    "standard_time": standard_time_discretizer,
    "kbins": kbins_discretizer,
    "standard_scaler": standard_scaler
}