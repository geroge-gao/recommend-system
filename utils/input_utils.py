
# -*- coding:utf-8 -*-
"""
Describe: input layer function for tf 1.x

Author: gerogegao

Date: 2022-08-22
"""

import sys
sys.path.append("./")

import tensorflow as tf
from .config import feat_spec


def build_input_layer():
    input_layer = {}

    for input_name in feat_spec["inputs"]:
        input_type = feat_spec["inputs"][input_name]
        if input_type == 'int':
            input_feature = tf.keras.Input((), name=input_name, dtype=tf.int32)
        elif input_type == 'float':
            input_feature = tf.keras.Input((), name=input_name, dtype=tf.float32)
        elif input_type == 'str':
            input_feature = tf.keras.Input((), name=input_name, dtype=tf.string)
        elif input_type == "embed":
            input_feature = tf.keras.Input((feat_spec['embed_feat'][input_name]["input_dim"], ), name=input_name, dtype=tf.float32)
        elif input_type == "list":
            input_feature = tf.keras.Input((feat_spec["list_feat"][input_name], ), name=input_name, dtype=tf.int32)
        input_layer[input_name] = input_feature
    return input_layer


def build_feature_columns():
    num_feature_arr = []
    onehot_feature_arr = []
    embedding_feature_arr = []

    feature_column_spec = feat_spec['inputs']

    base_cate_map = {}
    for num_feature in feature_column_spec['num']:
        num_feature_arr.append(tf.feature_column.numeric_column(num_feature['feature']))
    for cate_feature in feature_column_spec['cate']:
        base_cate_map[cate_feature['feature'] + '#cate'] = (tf.feature_column.categorical_column_with_vocabulary_list(cate_feature['feature'], cate_feature['vocab']), cate_feature)
    # for hash_feature in feature_column_spec['hash']:
    #     base_cate_map[hash_feature['feature'] + '#hash'] = (tf.feature_column.categorical_column_with_hash_bucket(hash_feature['feature'], hash_feature['bucket']), hash_feature)
    # for bucket_feature in feature_column_spec['bucket']:
    #     num_feature = tf.feature_column.numeric_column(bucket_feature['feature'])
    #     base_cate_map[bucket_feature['feature'] + '#bucket'] = (tf.feature_column.bucketized_column(num_feature, boundaries=bucket_feature['boundaries']), bucket_feature)

    cross_cate_map = {}
    if "cross" in feature_column_spec:
        for cross_feature in feature_column_spec['cross']:
            cols = []
            for col_name in cross_feature['feature']:
                column, spec = base_cate_map[col_name]
                cols.append(column)
            cross_cate_map['&'.join(cross_feature['feature']) + '#cross'] = tf.feature_column.crossed_column(cols, hash_bucket_size=cross_feature['bucket'])

    for cate_name in base_cate_map:
        column, spec = base_cate_map[cate_name]
        # print(base_cate_map)
        # onehot_feature_arr.append(tf.feature_column.indicator_column(column))
        embedding_feature_arr.append(tf.feature_column.embedding_column(column, spec['embedding']))
    # for cross_cate_name in cross_cate_map:
    #     cross_feature_col = cross_cate_map[cross_cate_name]
    #     onehot_feature_arr.append(tf.feature_column.indicator_column(cross_feature_col))

    return num_feature_arr + embedding_feature_arr
