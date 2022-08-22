# -*- coding:utf-8 -*-
"""
Describe: some method to parse tfrecord file
"""

import pandas as pd
from copy import deepcopy
import tensorflow as tf
from utils.config import feat_spec


def dataframe_to_tfrecord(df, tfrecord_filename):
    """convert data format from dataframe to tfrecord """
    feat_dict = {}
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for idx, d in df.iterrows():
            for f in feat_spec["inputs"]:
                feature_value = d[f]
                if feat_spec["inputs"][f] == "int":
                    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(feature_value)]))
                elif feat_spec[f] == "float":
                    feature = tf.train.Feature(float_list=tf.train.FloatList(value=[float(feature_value)]))
                elif feat_spec[f] == "object":
                    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(feature_value).encode("utf-8")]))
                feat_dict[f] = feature
            features = tf.train.Features(feature=feat_dict)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def get_feature_dict():
    """define tfrecord feature dict"""
    feature_dict = {}
    for feature_name in feat_spec["inputs"]:
        feature = None
        if feat_spec["inputs"][feature_name] == 'int':
            feature = tf.io.FixedLenFeature([], tf.int64)
        elif feat_spec["inputs"][feature_name] == 'float':
            feature = tf.io.FixedLenFeature([], tf.float32)
        elif feat_spec["inputs"][feature_name] == "list":
            feature = tf.io.FixedLenFeature([feat_spec["embed_feat_dim"][feature_name]], tf.float32)
      
        if feature:
            feature_dict[feature_name] = feature

    return feature_dict


def parse_func(ds, feature_dict, include_outputs=True):
    """parse single data from tfrecord"""
    inputs = tf.io.parse_single_example(ds, feature_dict)
    outputs = []
    if include_outputs:
        for output_name in feat_spec['outputs']:
            outputs.append(inputs[output_name])
            inputs.pop(output_name)
    if include_outputs:
        return inputs, outputs
    return inputs


def parse_single_example(dataset):

    example = tf.train.Example()
    ds = dataset.take(1)
    feature_dict = example.ParseFromString(ds.numpy())
    feature_type = feature_dict["features"]

    feature_names = []
    data_type = []

    for feat in feature_type:
        feat_example = feat["feature"]
        feature_names.append(feat_example["key"])
        
        if feat_example["value"] == "int64_list":
            data_type.append("int")

        elif feat_example["value"] == "float_list":
            data_type.append("float")

    type_frame = pd.DataFrame({"feature_name": feature_names, "feature_type": data_type})
    
    return type_frame


def get_hdfs_from_tfrecord(listdir):
    files = tf.data.Dataset.list_files(listdir)
    type_name = parse_single_example(files)
    return type_name



    
