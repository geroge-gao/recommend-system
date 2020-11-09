# -*- coding: utf-8 -*-

from tqdm import tqdm
import time
import numpy as np
import math
from collections import defaultdict


class UserCF:
    """
        recommend items from people around you
    """

    def __init__(self, data, rec_nums):
        self.data = data
        self.rec_nums = rec_nums
        self.user2item = {}
        self.item2user = {}
        self.user_sim = {}

    def get_top_items(self):
        hot_items = list(self.data['item_id'].value_counts().index)
        return hot_items[0: self.rec_nums]

    def user_similarity(self):
        # calculate similarity of users

        print('start to calculate user similarity')
        start_time = time.time()
        self.user2item = dict(self.data.groupby(['user_id'])['item_id'].apply(lambda x: list(x)[::-1]))
        self.item2user = dict(self.data.groupby(['item_id'])['user_id'].apply(lambda x: list(x)))


        user_similarity = {}
        c = {}
        w = {}
        for i, users in tqdm(self.item2user.items()):
            for u in users:
                c.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    c[u][v] += 1 / math.log(1 + len(users))

        for u, related_users in tqdm(c.items()):
            w.setdefault(u, defaultdict(int))
            for v, count in related_users.items():
                w[u][v] = count / math.sqrt(len(self.user2item[u]) * len(self.user2item[v]))

        end_time = time.time()
        print('finish calculating user similarity, it costs {} seconds'.format(end_time - start_time))
        self.user_sim = w

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
        if len(result) < self.rec_nums:
            tmp = [a for a in hot_items if a not in result]
            result = result + tmp[:(self.rec_nums - len(result))]
        return result

    def predict(self, user_list):
        # recommend item list for all users
        result = {}
        for user in user_list:
            result[user] = self.recommend(user)
        return result








