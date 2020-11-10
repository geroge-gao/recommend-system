import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
import tensorflow.keras
import tensorflow as tf
from recall_model.ncf import NCF

import scipy.sparse as sp
import os
import numpy as np
from utils.dataset_utils import split_train_test_data
from prepare_data import movielens
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop

size = '1m'
local_path = './data/ml-{}'.format(size)
data = movielens.load_pandas_df('1m',
                                ['user_id', 'item_id', 'rating', 'timestamp'],
                                title_col='title',
                                genres_col='genres',
                                year_col='year',
                                local_cache_path=local_path
                                )

data = data.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])

n_user = data['user_id'].nunique()
n_item = data['item_id'].nunique()


train, test = split_train_test_data(data)

ncf = NCF(n_users=n_user,
          n_items=n_item,
          model_type='mlp')

train_data = ncf.get_train_instance(train, num_negatives=1)

user_input = list(train_data['user_id'].values)
item_input = list(train_data['item_id'].values)
labels = list(train_data['labels'])

train_data = [np.array(user_input), np.array(item_input)]
labels = np.array(labels)

ncf.train(train_data, labels)

test_data = ncf.get_train_instance(test, num_negatives=1)




