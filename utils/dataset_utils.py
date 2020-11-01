import datetime
import time


def split_train_test_data(data):
    # use the last interactive item of each user for test
    train = data.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    test = data.groupby('user_id').tail(1)
    return train, test


