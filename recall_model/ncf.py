# -*- coding: utf-8 -*-

import tqdm
import time
import tensorflow as tf
from tensorflow.keras.layers import (Embedding,
                                     Input,
                                     Dense,
                                     Reshape,
                                     Flatten,
                                     Multiply,
                                     Concatenate)

from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras import initializers, Model
from tensorflow.keras.regularizers import l2


class NCF:
    """
    neural collaborative filtering: use mlp to replace
    Link: https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
    """
    def __init__(self,
                 n_users=10,
                 n_items=10,
                 model_type="NeuMF",
                 user_embedding_dim=10,
                 item_embedding_dim=10,
                 n_factors=8,
                 layer_sizes=[16, 8, 4]):
        """
        init parameters of the model
        :param n_users: numbers of user in the dataset
        :param n_items:
        :param model_type:
        :param n_factors:
        :param layer_sizes:
        """
        self.n_users = n_users
        self.n_items = n_items
        self.model_type = model_type
        self.n_factors = n_factors
        self.layer_size = layer_sizes
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim

    def get_model(self):

        try:
            if self.model_type == 'mlp':
                model = self.MLP()
            elif self.model_type == 'gmf':
                model = self.GMF()
            elif self.model_type == 'neuMF':
                model = self.NeuMF()
            else:
                raise ValueError("model_type must in ['mlp', 'gmf', 'neuMF']")
        except ValueError as e:
            print(repr(e))
        return model



    def MLP(self, layers_unit=[20, 10], mlp_reg=0):

        # define input layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        print(layers_unit)
        # define embedding function
        user_embedding = Embedding(input_dim=self.n_users,
                                   output_dim=self.user_embedding_dim,
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=l2(mlp_reg),
                                   input_length=1,
                                   name='user_embedding')(user_input)

        item_embedding = Embedding(input_dim=self.n_items,
                                   output_dim=self.item_embedding_dim,
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=l2(mlp_reg),
                                   input_length=1,
                                   name='item_embedding')(item_input)

        # fatten an embedding vector
        user_latent = Flatten()(user_embedding)
        item_latent = Flatten()(item_embedding)

        # concat embedding vector of user and item
        vector = Concatenate(axis=-1)([user_latent, item_latent])

        # mlp layers
        for i in range(len(layers_unit)):
            vector = Dense(layers_unit[i], activation='relu', name='mlp_layer-%d' % i)(vector)

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

        # Build mlp model
        model = Model(inputs=[user_input, item_input], outputs=prediction)

        return model

    def GMF(self, layers_unit=[20, 10], mlp_reg=0):

        # Define input layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # Define embedding layer
        user_embedding = Embedding(input_dim=self.n_users,
                                   output_dim=layers_unit[1],
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=l2(mlp_reg),
                                   input_length=1,
                                   name='user_embedding')(user_input)

        item_embedding = Embedding(input_dim=self.n_items,
                                   output_dim=layers_unit[1],
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=l2(mlp_reg),
                                   input_length=1,
                                   name='item_embedding')(item_input)

        # Fatten an embedding vector
        user_latent = Flatten()(user_embedding)
        item_latent = Flatten()(item_embedding)

        # Element-wise product of user and item embeddings
        predict_vector = Multiply()([user_latent, item_latent])

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)

        # Build gmf model
        model = Model(inputs=[user_input, item_input],
                      outputs=prediction)
        return model

    def NeuMF(self, layers_unit=[20, 10], reg_unit=0, mlp_reg=0):

        # define input layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        print(layers_unit)
        # define mlp embedding layer
        mlp_user_embedding = Embedding(input_dim=self.n_users,
                                       output_dim=layers_unit[0],
                                       embeddings_initializer='uniform',
                                       embeddings_regularizer=l2(mlp_reg),
                                       input_length=1,
                                       name='mlp_user_embedding')(user_input)

        mlp_item_embedding = Embedding(input_dim=self.n_items,
                                       output_dim=layers_unit[0],
                                       embeddings_initializer='uniform',
                                       embeddings_regularizer=l2(mlp_reg),
                                       input_length=1,
                                       name='mlp_item_embedding')(item_input)

        # define gmf embedding layer
        gmf_user_embedding = Embedding(input_dim=self.n_users,
                                       output_dim=layers_unit[1],
                                       embeddings_initializer='uniform',
                                       embeddings_regularizer=l2(mlp_reg),
                                       input_length=1,
                                       name='user_embedding')(user_input)

        gmf_item_embedding = Embedding(input_dim=self.n_items,
                                       output_dim=layers_unit[1],
                                       embeddings_initializer='uniform',
                                       embeddings_regularizer=l2(mlp_reg),
                                       input_length=1,
                                       name='item_embedding')(item_input)

        # fatten mlp embedding vector
        mlp_user_latent = Flatten(name='mlp_user_flatten')(mlp_user_embedding)
        mlp_item_latent = Flatten(name='mlp_item_flatten')(mlp_item_embedding)

        # concat embedding vector of user and item
        mlp_vector = Concatenate(axis=-1, name='mlp_vector')([mlp_user_latent, mlp_item_latent])

        # mlp layers
        for i in range(len(layers_unit)):
            mlp_vector = Dense(layers_unit[i], activation='relu', name='mlp_layer-%d' % i)(mlp_vector)

        # fatten gmf embedding vector
        gmf_user_latent = Flatten(name='gmf_user_flatten')(gmf_user_embedding)
        gmf_item_latent = Flatten(name='gmf_item_flatten')(gmf_item_embedding)

        # multiply gmf user embedding and gmf item embedding
        gmf_vector = Multiply(name='gmf_vector')([gmf_user_latent, gmf_item_latent])

        # concat gmf  and mlp parts
        predict_vector = Concatenate(axis=-1)([mlp_vector, gmf_vector])

        # final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='neuMF_prediction')(predict_vector)

        # Build neuMF model
        model = Model(inputs=[user_input, item_input], outputs=prediction)

        return model

    def fit(self, model, data):
        # compile the model
        print('fit the model')
        self.model_type = 1

    def load_pretrain_model(self, model, gmf_model_path, mlp_model_path):
        self.n_users = 1

    def recommend(self, model, path):
        print('recommend function')
        self.n_users = 1







