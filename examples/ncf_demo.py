import numpy as np
import os
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

# define mlp model

mlp = NCF(n_users=n_user,
          n_items=n_item,
          learning_rate=0.001,
          user_mlp_embedding_dim=10,
          item_mlp_embedding_dim=10,
          layers=[20, 10],
          optimizer='adam',
          model_type='mlp',
          verbose=1)

# get positive and negative samples of train set
train_data = mlp.get_train_instance(train, num_negatives=1)

user_input = list(train_data['user_id'].values)
item_input = list(train_data['item_id'].values)
labels = list(train_data['label'].values)

# fit the input data
inputs = [np.array(user_input), np.array(item_input)]
labels = np.array(labels)

# get test data
test_data = test[['user_id', 'item_id']]
test_data['label'] = 1

# mlp train
print('start to train')
mlp.train(inputs, labels)

# test_data
test_user = list(test_data['user_id'].unique())
test_item = list(test_data['item_id'].unique())
test_labels = test_data['label'].values

# mlp predict
mlp_prediction = mlp.predict(test_user, test_item, top_k=100)

# mlp evaluate
print("mlp ndcg: {}".format(ndcg(test_data, mlp_prediction)))
print("mlp hit rate: {}".format(hit_ratio(test_data, mlp_prediction)))

# save mlp model
mlp_model_dir = './model/ncf'
mlp_file = 'mlp.h5'
mlp.save_model(mlp_model_dir)

# define gmf model
gmf = NCF(n_user,
          n_item,
          model_type='gmf',
          learning_rate=0.001,
          user_gmf_embedding_dim=10,
          item_gmf_embedding_dim=10,
          layers=[20, 10],
          reg_layers=[0, 0],
          optimizer='adam',
          epochs=1,
          batch_size=32,
          verbose=1)

# gmf train
gmf.train(inputs, labels)

# gmf predict
gmf_prediction = gmf.predict(test_user, test_item, top_k=100)

# gmf evaluation
print("gmf ndcg: {}".format(ndcg(test_data, gmf_prediction)))
print("gmf hit rate: {}".format(hit_ratio(test_data, gmf_prediction)))

# save gmf
gmf_model_path = './model/ncf/'
gmf_file = 'gmf.h5'
gmf.save_model(gmf_model_path)

# define neuMF
neuMF = NCF(n_user,
            n_item,
            model_type='neuMF',
            learning_rate=0.001,
            user_mlp_embedding_dim=10,
            item_mlp_embedding_dim=10,
            user_gmf_embedding_dim=10,
            item_gmf_embedding_dim=10,
            optimizer='adam',
            layers=[20, 10],
            batch_size=32,
            verbose=1,
            load_pretrain=True
            )

# train neuMF
mlp_path = os.path.join(mlp_model_dir, mlp_file)
gmf_path = os.path.join(gmf_model_path, gmf_file)
neuMF.train(inputs, labels, mlp_dir=os.path.realpath(mlp_path), gmf_dir=os.path.realpath(gmf_path))

# neuMF predict
neuMF_prediction = neuMF.predict(test_user, test_item, top_k=100)

# neuMF evaluation
print("neuMF ndcg: {}".format(ndcg(test_data, neuMF_prediction)))
print("neuMF hit rate: {}".format(hit_ratio(test_data, neuMF_prediction)))




