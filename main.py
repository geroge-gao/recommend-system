# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
from prepare_data import movielens
import tensorflow as tf
from utils.newsrec_utils import get_mind_data_set, prepare_hparams
from utils.deeprec_utils import download_deeprec_resources
from tempfile import TemporaryDirectory

pd.set_option('display.max_columns', None)
sys.path.append('../../')


# load the movies data
# size: 100k, 1m ,10m, 20m

size = '1m'
local_path = './data/ml-{}'.format(size)

# data = movielens.load_pandas_df('1m',
#                                 ['UserId', 'ItemId', 'Rating', 'Timestamp'],
#                                 title_col='Title',
#                                 genres_col='Genres',
#                                 year_col='Year',
#                                 local_cache_path=local_path
#                                 )
# print(data.head(5))

epochs = 5
seed = 40
MIND_type = 'large'
tmpdir = TemporaryDirectory()
data_path = tmpdir.name
data_path = './data/mind-{}'.format(MIND_type)

# def get_mind_dataset(datapath):
train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'lstur.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)

if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/', \
                               os.path.join(data_path, 'utils'), mind_utils)

# hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, \
#                           wordDict_file=wordDict_file, userDict_file=userDict_file, epochs=epochs)
# print(hparams)


