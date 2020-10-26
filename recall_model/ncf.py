# -*- coding: utf-8 -*-

import tqdm
import time
import tensorflow as tf
from tensorflow.python.keras.layers import (Embedding,
                                            Input,
                                            Dense,
                                            Reshape,
                                            Flatten,
                                            merge)

from tensorflow.python.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.python.keras import initializers
from tensorflow.python.keras.regularizers import l2




class NCF:
    """
    neural collaborative filtering: use mlp to replace
    Link: https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
    """
    def __init__(self,
                 n_users,
                 n_items,
                 model_type="NeuMF",
                 user_embedding_dim=10,
                 item_embedding_dim=10,
                 n_factors=8,
                 layer_sizes=[16, 8, 4],
                 n_epochs=50,
                 batch_size=64,
                 learning_rate=5e-3,
                 verbose=1,
                 seed=None):
        """
        construct function
        :param n_users: numbers of user in the dataset
        :param n_items:
        :param model_type:
        :param n_factors:
        :param layer_sizes:
        :param n_epochs:
        :param batch_size:
        :param learning_rate:
        :param verbose:5
        :param seed:
        """
        self.n_users = n_users
        self.n_items = n_items
        self.model_type = model_type
        self.n_factors = n_factors
        self.layer_size = layer_sizes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def MLP(self, n_user, n_items, user_embedding_dim, item_embedding_dim, layers, mlp_reg=0):
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # define embedding function
        user_embedding = Embedding(input_dim=self.n_users, outpuit_dim=user_embedding_dim,
                                   name='user_embedding', embeddings_initializer=initializers.normal((1,user_embedding_dim),
                                   scale=0.01), W_regularizer=l2(mlp_reg), input_length=1)(user_input)

        item_embedding = Embedding(input_dim=self.n_items, output_dim=item_embedding_dim,
                                   name='item embedding', embeddings_initializer=initializers.normal((1, item_embedding_dim), scale=0.01),
                                   embeddings_regularizer=l2(mlp_reg), input_length=1)(item_input)

        # fatten an embedding vector
        user_latent = Flatten()(user_embedding)
        item_latent = Flatten(item_embedding)

        # concat embedding vector of user and item
        vector = merge([user_latent, item_latent], mode='concat')

        #


        return model
    def GMF(self):
        self.n_factors = 1
        return 1

    def NeuMF(self):

        return 1

    def create_model(self):
        # input_layer = k.layers.InputLayer()

    def fit(self, data):
        self.data = data

    def save(self, dir_name):
        # save the model
        self.model_type = 11
        return 1





