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
        init parameters of the model
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
        self.
        self.seed = seed

    def MLP(self, layers_unit=[20,10], mlp_reg=0):
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # define embedding function
        user_embedding = Embedding(input_dim=self.n_users, outpuit_dim=layers_unit[1],
                                   name='user_embedding', embeddings_initializer=initializers.normal((1,layers_unit[1]),
                                   scale=0.01), W_regularizer=l2(mlp_reg), input_length=1)(user_input)

        item_embedding = Embedding(input_dim=self.n_items, output_dim=layers_unit[1],
                                   name='item embedding', embeddings_initializer=initializers.normal((1, layers_unit[1]), scale=0.01),
                                   embeddings_regularizer=l2(mlp_reg), input_length=1)(item_input)

        # fatten an embedding vector
        user_latent = Flatten()(user_embedding)
        item_latent = Flatten(item_embedding)

        # concat embedding vector of user and item
        vector = merge([user_latent, item_latent], mode='concat')

        # mlp layers
        for i in range(len(layers_unit)):
            vector = Dense(layers_unit[i], activation='relu')(vector)


    def GMF(self):
        self.n_factors = 1
        return 1

    def NeuMF(self):

        return 1


    def fit(self, data):
        self.data = data

    def save(self, dir_name):
        # save the model
        self.model_type = 11
        return 1





