from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from utils.dataset_utils import load_adult_data
from rank_model.deep_wide import Deep_Wide

from tensorflow.keras.utils import plot_model

deep_wide = Deep_Wide()
wide = deep_wide.wide_model([1,2,3])

train, test = load_adult_data("./data/adult/")

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation", "relationship",
    "race", "gender", "native_country"
]

cate_columns = []
cate_nums = []
for i in CATEGORICAL_COLUMNS:
    cate_columns.append(i)
    cate_nums.append(train[i].nunique())

col_nums = dict(zip(cate_columns, cate_nums))


CONTINUOUS_COLUMNS = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]

deep = deep_wide.deep_model(col_nums, CONTINUOUS_COLUMNS)

from utils.dataset_utils import category_cross_feature
train = train.dropna(how='any',axis=0)
data = category_cross_feature(train[cate_columns], degree=2)
cross_nums = data.shape[1]

cross_feature = [i for i in range(cross_nums)]

wide_deep = deep_wide.deep_wide(col_nums, CONTINUOUS_COLUMNS, cross_feature)