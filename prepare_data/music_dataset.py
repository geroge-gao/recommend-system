import pandas as pd


def get_pandas_df(path):
    """
    load Ali music DataFrame
    :param path: the local path of DataSet
    :return data:
    """

    data = pd.read_csv(path)
    return data
