import numpy as np
import math


def hit_ratio(test, prediction):
    """
    calculate mean hit ratio
    :param test:
    :param prediction:
    :return: average hit ratio of each user
    """

    prediction = prediction.merge(test, on=['user_id', 'item_id'], how='left')
    prediction['label'] = prediction['label'].fillna(0)
    hit_data = prediction.groupby('user_id')['label'].agg({'total': 'sum'})
    hit_data['is_hit'] = hit_data['total'].apply(lambda x: 0 if x == 0 else 1)

    return hit_data['is_hit'].sum() / len(hit_data)


def ndcg(test, prediction):
    """
    calcuate mean ndcg
    :param test: test data: [user_id, item_id, label]
    :param prediction: prediction result: [user_id, item_id, score, rank]
    :return: average ndcg of each user
    """
    prediction = prediction.merge(test, on=['user_id', 'item_id'], how='left')
    prediction = prediction[prediction['label'].isna() == False]
    prediction['ndcg'] = prediction['rank'].apply(lambda x: math.log(2)/math.log(x+1))
    return prediction['ndcg'].sum() * 1.0 / prediction['user_id'].nunique()
