# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
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
from tensorflow.keras.models import load_model



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
                 learning_rate=10e3,
                 layers=[16, 8, 4],
                 epochs=100,
                 optimizer='adam',
                 loss='binary_crossentropy',
                 batch_size=32,
                 verbose=0):
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
        self.layers = layers
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss

        self.model = self.get_model()

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

    def MLP(self, layers=[20, 10], mlp_reg=0):

        # define input layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        print(layers)
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
        for i in range(len(layers)):
            vector = Dense(layers[i], activation='relu', name='mlp_layer-%d' % i)(vector)

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

        # Build mlp model
        model = Model(inputs=[user_input, item_input], outputs=prediction)

        return model

    def GMF(self, layers=[20, 10], mlp_reg=0):

        # Define input layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # Define embedding layer
        user_embedding = Embedding(input_dim=self.n_users,
                                   output_dim=layers[1],
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=l2(mlp_reg),
                                   input_length=1,
                                   name='user_embedding')(user_input)

        item_embedding = Embedding(input_dim=self.n_items,
                                   output_dim=layers[1],
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

    def NeuMF(self, layers=[20, 10], reg_unit=0, mlp_reg=0):

        # define input layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        print(layers)
        # define mlp embedding layer
        mlp_user_embedding = Embedding(input_dim=self.n_users,
                                       output_dim=layers[0],
                                       embeddings_initializer='uniform',
                                       embeddings_regularizer=l2(mlp_reg),
                                       input_length=1,
                                       name='mlp_user_embedding')(user_input)

        mlp_item_embedding = Embedding(input_dim=self.n_items,
                                       output_dim=layers[0],
                                       embeddings_initializer='uniform',
                                       embeddings_regularizer=l2(mlp_reg),
                                       input_length=1,
                                       name='mlp_item_embedding')(item_input)

        # define gmf embedding layer
        gmf_user_embedding = Embedding(input_dim=self.n_users,
                                       output_dim=layers[1],
                                       embeddings_initializer='uniform',
                                       embeddings_regularizer=l2(mlp_reg),
                                       input_length=1,
                                       name='user_embedding')(user_input)

        gmf_item_embedding = Embedding(input_dim=self.n_items,
                                       output_dim=layers[1],
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
        for i in range(len(layers)):
            mlp_vector = Dense(layers[i], activation='relu', name='mlp_layer-%d' % i)(mlp_vector)

        # fatten gmf embedding vector
        gmf_user_latent = Flatten(name='gmf_user_flatten')(gmf_user_embedding)
        gmf_item_latent = Flatten(name='gmf_item_flatten')(gmf_item_embedding)

        # multiply gmf user embedding and gmf item embedding
        gmf_vector = Multiply(name='gmf_vector')([gmf_user_latent, gmf_item_latent])

        # concat gmf  and mlp parts
        predict_vector = Concatenate(axis=1)([mlp_vector, gmf_vector])

        # final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='neuMF_prediction')(predict_vector)

        # Build neuMF model
        model = Model(inputs=[user_input, item_input], outputs=prediction)

        return model

    def get_train_instance(self, data, num_negatives):
        """
        construct positive and negative samples for training set
        :param data: train data of which the data format is  [user_id, item_id,...]
        :param num_negatives: number of negative instances to pair with a positive instance
        :return:
        """

        # get the positive data
        positive_data = data.iloc[:, 0:2]
        positive_data['labels'] = 1

        user_input, item_input, labels = [], [], []
        user2item_list = positive_data.groupby('user_id')['item_id'].agg(lambda x: list(x)).reset_index()
        mat = dict(zip(user2item_list['user_id'].values, user2item_list['item_id'].values))
        for u in mat:
            length = len(mat[u])
            for i in range(length):
                for n in range(num_negatives):
                    j = np.random.randint(self.n_items)
                    while j in mat[u]:
                        j = np.random.randint(self.n_items)
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
        # get negative data
        negative_data = pd.DataFrame({'user_id': user_input, 'item_id': item_input, 'labels': labels})

        # concat positive and negative data
        train = pd.concat([positive_data, negative_data]).reset_index(drop=True)
        train = shuffle(train)

        return train

    def train(self, data, labels, split_ratio=0.2):
        # compile and train the model
        print('fit the model')

        # user_input = list(data['user_id'].values)
        # item_input = list(data['item_id'].values)
        # labels = list(data['labels'].values)

        if self.optimizer.lower() == 'adagrad':
            self.model.compile(optimizer=Adagrad(learning_rate=self.learning_rate), loss=self.loss)
        elif self.optimizer.lower() == 'adam':
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)
        elif self.optimizer.lower() == 'rmsprop':
            self.model.compile(optimizer=RMSprop(self.learning_rate), loss=self.loss)
        else:
            self.model.compile(optimizer=SGD(self.learning_rate), loss=self.loss)

        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.fit(data,
                           labels,
                           verbose=self.verbose,
                           epochs=1,
                           batch_size=self.batch_size,
                           validation_split=split_ratio,
                           shuffle=True)

            end_time = time.time()
            print(end_time - start_time)

    def load_pretrain_model(self, gmf_model_path, mlp_model_path):
        # check whether the model file exists
        if os.path.exists(gmf_model_path) and os.path.exists(mlp_model_path):
            print('the model path is not correct, please check whether the model files exist')
        else:
            # load model weights
            gmf_model = load_model(gmf_model_path)
            mlp_model = load_model(mlp_model_path)

            # MF embeddings
            gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
            gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
            self.model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
            self.model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

            # MLP embeddings
            mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
            mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
            self.model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
            self.model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

            # MLP layers
            for i in range(1, self.layers):
                mlp_layer_weights = mlp_model.get_layer('mlp_layer-%d' % i).get_weights()
                self.get_layer('mlp_layer-%d' % i).set_weights(mlp_layer_weights)

            # Prediction weights
            gmf_prediction = gmf_model.get_layer('prediction').get_weights()
            mlp_prediction = mlp_model.get_layer('prediction').get_weights()
            new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
            new_b = gmf_prediction[1] + mlp_prediction[1]
            print('new_b.shape', new_b.shape)
            self.model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])

    def save_model(self, directory):
        """
        save the model
        :param directory:
        :param dir: the directory where to put the model
        :return:
        """

        # create the dir if not exist
        if not os.path.exists(directory):
            os.mkdir(directory)

        # remove the model file if exists
        model_name = self.model_type + '.h5'
        model_path = os.join(directory, model_name)
        if os.path.exists(model_path):
            os.remove(model_path)

        self.model.save(model_path)

    def recommend(self, users, items, top_k):
        print('recommend function')

        result = {}
        # for u in users
        for u in users:
            user = np.full(self.n_items, u, dtype='int32')
            prediction = self.model.predict([np.array(user), np.array(items)])
            # get top k item
            item_scores = dict(zip(items, prediction))
            top_items = list(sorted(item_scores.items(), key=lambda x: x[1], reverse=True))[:top_k]
            result[u] = top_items

        return result




            









