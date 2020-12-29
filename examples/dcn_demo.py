import pandas as pd
from rank_model.deep_cross import DCN
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

train = pd.read_csv('../data/higgs-boson/training.csv')
test = pd.read_csv('../data/higgs-boson/test.csv')

columns = list(train.columns.values)
sparse_features = ['PRI_jet_num']
dense_features = [i for i in columns if i not in ['Label', 'PRI_jet_num', 'Weight']]

sparse_columns = dict(zip(list(train[sparse_features].nunique().index), list(train[sparse_features].nunique().values)))
dense_columns = dense_features

for feat in sparse_features:
    le = LabelEncoder()
    train[feat] = le.fit_transform(train[feat])
    test[feat] = le.transform(test[feat])

train['Label'] = LabelEncoder().fit_transform(train['Label'])


mms = MinMaxScaler(feature_range=(0, 1))
train[dense_features] = mms.fit_transform(train[dense_features])
test[dense_features] = mms.fit_transform(test[dense_features])

# transform the dataframe to numpy
dense_input_train = np.array(train[dense_features], dtype='float32')
sparse_input_train = np.array(train[sparse_features], dtype='int32')
sparse_input_train = [sparse_input_train[:, i] for i in range(len(sparse_features))]

dense_input_test = np.array(test[dense_features], dtype='float32')
sparse_input_test = np.array(test[sparse_features], dtype='int32')
sparse_input_test = [sparse_input_test[:, i] for i in range(len(sparse_features))]

train_input = [dense_input_train] + sparse_input_train
train_labels = np.array(train['Label'], dtype='int32')

# init deep cross parameter
lr = 0.001
op = 'adam'
batch_size = 512
batch_size = 4
latent = 4
seed = 2020
layers = [32, 64, 128, 256, 1024]
layers_reg = [0, 0, 0, 0, 0]
loss_func = 'sparse_categorical_crossentropy'

# define deepFM model
model = DCN(dense_columns=dense_columns,
            sparse_columns=sparse_columns,
            dnn_layers=layers,
            cross_layer_nums=2,
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

# test








