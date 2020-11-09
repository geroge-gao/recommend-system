import numpy as np
from recall_model.ncf import NCF
from prepare_data import movielens
from evaluate.evaluation import ndcg, hit_ratio
from utils.dataset_utils import split_train_test_data
import warnings
warnings.filterwarnings("ignore")

# load dataset
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

# split train and test
train, test = split_train_test_data(data)

# get user numbers and item numbers
n_user = train['user_id'].nunique()
n_item = train['item_id'].nunique()

# define the mlp model
#
ncf = NCF(n_users=n_user,
          n_items=n_item,
          learning_rate=0.001,
          user_mlp_embedding_dim=10,
          item_mlp_embedding_dim=10,
          optimizer='adam',
          model_type='mlp')

# get positive and negative samples of train set
train_data = ncf.get_train_instance(train, num_negatives=1)

user_input = list(train_data['user_id'].values)
item_input = list(train_data['item_id'].values)
labels = list(train_data['label'].values)

# fit the input data
inputs = [np.array(user_input), np.array(item_input)]
labels = np.array(labels)

# get test data
test_data = test[['user_id', 'item_id']]
test_data['label'] = 1

ncf.train(inputs, labels, split_ratio=0.1)

# test_data
test_user = list(test_data['user_id'].unique())
test_item = list(test_data['item_id'].unique())
test_labels = test_data['label'].values

# predict
recommendation = ncf.predict(test_user, test_item, top_k=100)

# evaluate
print("ndcg: {}".format(ndcg(test_data, recommendation)))
print("hit rate: {}".format(hit_ratio(test_data, recommendation)))

# save model
model_dir = './model/ncf'
ncf.save_model(model_dir)




