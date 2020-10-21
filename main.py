# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
from prepare_data import movielens
import tensorflow as tf
from prepare_data import music_dataset
from utils.newsrec_utils import get_mind_data_set, prepare_hparams
from utils.deeprec_utils import download_deeprec_resources

pd.set_option('display.max_columns', None)
sys.path.append('../../')


# load the movies data
# size: 100k, 1m ,10m, 20m

size = '1m'
local_path = './data/ml-{}'.format(size)

data = movielens.load_pandas_df('1m',
                                ['user_id', 'item_id', 'rating', 'timestamp'],
                                title_col='title',
                                genres_col='genres',
                                year_col='year',
                                local_cache_path=local_path
                                )

from recall_model.user_cf import UserCF

data = data.iloc[0:10000, :]

print(data.head())

user_cf = UserCF(data, rec_nums=50)
user_cf.user_similarity()
print(user_cf.recommend(1))
# print(user_cf.get_top_items())
# print(user_cf.user_sim)



# epochs = 5
# seed = 40
# MIND_type = 'demo'
# # tmpdir = TemporaryDirectory()
# # data_path = tmpdir.name
# data_path = os.path.join('./data', 'mind-{}'.format(MIND_type))
# print(data_path)
#
# # def get_mind_dataset(datapath):
# train_news_file = os.path.join(data_path, 'train', r'news.tsv')
# train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
# valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
# valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
# wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
# userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
# wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
# yaml_file = os.path.join(data_path, "utils", r'lstur.yaml')
#
# mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)
#
# if not os.path.exists(train_news_file):
#     download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
#
# if not os.path.exists(valid_news_file):
#     download_deeprec_resources(mind_url, os.path.join(data_path, 'valid'), mind_dev_dataset)
# if not os.path.exists(yaml_file):
#     download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/',\
#                                os.path.join(data_path, 'utils'), mind_utils)
#
#
# # load the news and behaviors data of mind data set.
# new_headers = ['news_id',
#                'category',
#                'subcategory',
#                'title',
#                'abstract',
#                'url',
#                'title_entities',
#                'abstract_entities']
# data = pd.read_csv(train_news_file, sep='\t', names=new_headers)
#
# behaviors_header = ['impression_id',
#                     'user_id',
#                     'time',
#                     'history',
#                     'impressions']
#
# behaviors_data = pd.read_csv(train_behaviors_file, sep='\t', names=behaviors_header)
#
# # get music data set
#
# local_path = './data/aliMusic/'
# action_file = 'p2_mars_tianchi_user_actions.csv'
# song_file = 'p2_mars_tianchi_songs.csv'
# action_path = os.path.join(local_path, action_file)
# song_path = os.path.join(local_path, song_file)
# action_df, song_df = music_dataset.get_pandas_df(action_file, song_path)

