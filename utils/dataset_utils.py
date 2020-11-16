import datetime
import time
import pandas as pd
import os
import numpy as np
from utils.download_utils import maybe_download
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder


def split_train_test_data(data):
    # use the last interactive item of each user for test
    train = data.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    test = data.groupby('user_id').tail(1)
    return train, test


def load_adult_data(workdir):

    # data set urls
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"
    # download data
    train_file = 'adult.data'
    test_file = 'adult.test'
    names_file = 'adult.names'
    train_path = maybe_download(train_url, train_file, workdir)
    test_path = maybe_download(test_url, test_file, workdir)

    # read local file
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "gender",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income_bracket"
    ]

    train = pd.read_csv(train_path, sep=',', names=columns)
    test = pd.read_csv(test_path, sep=',', names=columns)
    # names = pd.read_csv(names_path, sep=',')
    return train, test


def category_cross_feature(data, degree, interaction_only=True, include_bias=False):
    # create cross feature for DNN
    cross_features = list(data.columns.values)

    for col in cross_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    cross_feature = poly.fit_transform(data)
    return cross_feature









