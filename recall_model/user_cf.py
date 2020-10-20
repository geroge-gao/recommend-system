# -*- coding: utf-8 -*-

import warnings
from tqdm import tqdm
from datetime import datetime
import numpy as np
import math
from collections import defaultdict
warnings.filters('ignore')


class UserCF:
    """
        recommend items from people around you
    """

    def __init__(self, data, rec_nums):
        self.data = data
        self.rec_nums = rec_nums
        self.user2item = {}
        self.user_sim = {}

    def get_top_items(self):
        hot_items = list(self.data['item_id'].value_counts().index)
        return hot_items

    def user_similarity(self):
        # calculate similarity of users

        print('start to calculate user similarity')
        start_time = datetime.time()
        user2item = dict(self.data.groupby(['user_id'])['item_id'].apply(lambda x: list(x)[::-1]))
        item2user = dict(self.groupby(['item_id'])['user_id'].apply(lambda x: list(x)))

        user_sim = {}
        for i, users in tqdm(item2user.items()):
            for u in users:
                user_sim.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    user_sim[u][v] += 1 / math.log(1 + len(users))

        for u, related_users in tqdm(user_sim.items()):
            for v, count in related_users.items():
                user_sim[u][v] = count / math.sqrt(
                    len(user2item[u]) * len(user2item[v]))
        self.user_sim = user_sim
        end_time = datetime.time()
        print('finish calculating user similarity, it costs {} seconds'.format(end_time - start_time))

        return user_sim

    def recommend(self, user):
        # recommend item for specific user
        rank = dict()
        hot_items = self.get_top_items()
        interacted_items = self.user2item[user]
        for similar_user, sim_score in sorted(self.user_sim[user].items(), key=lambda x: x[1], reverse=True)[0:self.rec_nums]:
            for similar_user_interacted_item in self.user2item[similar_user]:
                if similar_user_interacted_item in interacted_items:
                    continue
                rank.setdefault(similar_user_interacted_item, 0)
                rank[similar_user_interacted_item] += sim_score
        result = list(dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:self.rec_nums]).keys())
        if len(result) < 50:
            tmp = [a for a in hot_items  if a not in result]
            result = result + tmp[:(50 - len(result))]
        return result

    def recommend_all(self, user_list):
        # recommend item list for all users
        result = {}
        for user in user_list:
            result[user] = self.recommend(user)
        return result








