import pandas as pd
from rank_model.deep_cross import DCN
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/crito/dac_sample.txt', sep='\t', header=None)
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]


columns = ['label'] + dense_features + sparse_features
data.columns = columns

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

train, test = train_test_split(data, test_size=0.2, random_state=2020)

sparse_columns = dict(zip(list(data[sparse_features].nunique().index), list(data[sparse_features].nunique().values)))
dense_columns = dense_features

# get train and test data
train[dense_features] = train[dense_features].astype('float32')
train[sparse_features] = train[sparse_features].astype('int32')
test[dense_features] = test[dense_features].astype('float32')
test[sparse_features] = test[sparse_features].astype('int32')

train_dense = train[dense_features].values
train_sparse = train[sparse_features].values
test_dense = test[dense_columns].values
test_sparse = test[sparse_features].values

train_sparse = [train_sparse[:, i] for i in range(train_sparse.shape[1])]
test_sparse = [test_sparse[:, i] for i in range(test_sparse.shape[1])]

train_input = [train_dense] + train_sparse
test_input = [test_dense] + test_sparse

train_labels = np.array(train['label'], dtype='int32')
test_labels = np.array(test['label'], dtype='int32')

# init deepFM parameter
lr = 0.001
op = 'adam'
batch_size = 512
batch_size = 10
latent = 4
seed = 2020
layers = [256, 128, 32, 1]
layers_reg = [0, 0, 0, 0]

# init deep cross parameter
lr = 0.001
op = 'adam'
batch_size = 512
batch_size = 4
latent = 4
seed = 2020
layers = [32, 64, 128, 256, 1024]
layers_reg = [0, 0, 0, 0, 0]
loss_func = 'binary_crossentropy'

# define deepFM model
model = DCN(dense_columns=dense_columns,
            sparse_columns=sparse_columns,
            dnn_layers=layers,
            cross_layer_nums=1,
            layer_reg=layers_reg,
            embed_dim=latent,
            seed=2020)


# train the model
model.train(x_train=train_input,
            y_train=train_labels,
            opt=op,
            learning=lr,
            model_loss=loss_func,
            batchsize=batch_size)








