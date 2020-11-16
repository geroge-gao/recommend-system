import pandas as pd
import numpy as np
from rank_model.deep_wide import Deep_Wide
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from utils.dataset_utils import load_adult_data, category_cross_feature


# load data
data_dir = '../data/adult'
train, test = load_adult_data(data_dir)

# different columns
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income_bracket"
]

continuous_columns = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]

category_columns = [
    "workclass", "education", "marital_status", "occupation", "relationship",
    "race", "gender", "native_country"
]

# encode category feature
# train = train.dropna(how='any', axis=0)
# test = train.dropna(how='any', axis=0)

train = train[train['income_bracket'].isna() == False]
test = test[test['income_bracket'].isna() == False]

data = pd.concat([train, test])
data = data.fillna('None')
for i in category_columns:
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])

print(data['income_bracket'].unique())

data['label'] = data['income_bracket'].apply(lambda x: x in ">50k").astype(int)
train = data.iloc[0:len(train), :]
test = data.iloc[len(train):, :]

# get cross feature for wide model
cross_fe_train = category_cross_feature(train[category_columns])
cross_fe_test = category_cross_feature(test[category_columns])

# get category feature
category_fe_train = np.array(train[category_columns], dtype='int32')
category_fe_test = np.array(test[category_columns], dtype='int32')

# transform array to list in column
categ_fe_train = [category_fe_train[:, i] for i in range(category_fe_train.shape[1])]
categ_fe_test = [category_fe_test[:, i] for i in range(category_fe_test.shape[1])]

# get continous feature
continous_fe_train = np.array(train[continuous_columns], dtype='float32')
continous_fe_test = np.array(test[continuous_columns], dtype='float32')


# define model
model_type = 'deep_wide'
layers = [512, 256, 128]
reg_layers = [0, 0, 0]

# define model input
train_input = None
test_input = None

# get model input
if model_type == 'wide':
    train_input = cross_fe_train
    test_input = cross_fe_test
elif model_type == 'deep':
    train_input = categ_fe_train + [continous_fe_train]
    input_test = categ_fe_test + [continous_fe_test]
elif model_type == 'deep_wide':
    train_input = [cross_fe_train, continous_fe_train] + categ_fe_train
    test_input = [cross_fe_test, continous_fe_test] + categ_fe_test

train_label = np.array(train['label'])
test_label = np.array(test['label'])

cross_nums = cross_fe_train.shape[1]
continous_nums = len(continuous_columns)

columns_count = {}
for i in category_columns:
    columns_count[i] = data[i].nunique()

# create deep&wide instance
deep_wide = Deep_Wide(model_type=model_type,
                      layers=layers,
                      reg_layers=reg_layers,
                      category_columns_count=columns_count,
                      continous_nums=continous_nums,
                      cross_nums=cross_nums)


# define training parameters
learning_rate = 0.001
epochs = 1
optimizer = 'adam'
loss = 'binary_crossentropy'
metric = 'accuracy'
lr = 0.001
epochs = 5
batch_size = 32
verbose = 1
validation_split = 0.1

# whether user pretrain model  or not, if use, we need to provide
# pretrain model path
load_pretrain_model = False
wide_dir = ''
deep_dir = ''

deep_wide.train(x_train=train_input,
                y_train=train_label,
                lr=learning_rate,
                opt=optimizer,
                loss_=loss,
                metric_=metric,
                n_epoch=epochs,
                n_batch_size=batch_size,
                validation_ratio=validation_split,
                n_verbose=verbose)

# evaluate model
deep_wide.evaluate_model(test_input, test_label)
prediction = deep_wide.predict(test_input)

# save model
model_dir = './model/wide_deep'
model_path = deep_wide.save_model(model_dir)
print(model_path)

# visualize the training logs
deep_wide.plot_picture()




















# define wide and deep model







