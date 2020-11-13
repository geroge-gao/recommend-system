from recall_model import item_cf
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
test['label'] = 1

# define item_cf model
itemCF = item_cf.ItemCF(train, rec_nums=100)

# calculate similarity of item
itemCF.item_similarity()

# get top-n item
itemCF.get_hot_item()

# predict
users = list(test['user_id'].unique())
prediction = itemCF.predict(users)

# itemCF evaluation
print("itemCF ndcg: {}".format(ndcg(test, prediction)))
print("itemCF hit rate: {}".format(hit_ratio(test, prediction)))

