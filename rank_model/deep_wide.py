import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Input,
                                     Dense,
                                     Embedding,
                                     Flatten,
                                     Concatenate,
                                     concatenate,
                                     BatchNormalization)

from tensorflow.keras.optimizers import SGD, Adagrad, Adam, RMSprop


class Deep_Wide:

    def __init__(self,
                 model_type="NeuMF",
                 learning_rate=0.1,
                 layers=[20, 10],
                 reg_layers=[0, 0],
                 epochs=1,
                 optimizer='adam',
                 loss='binary_crossentropy',
                 batch_size=32,
                 verbose=1):
        """
        init parameters of the model
        :param n_users: numbers of user in the dataset
        :param n_items:
        :param model_type:
        :param n_factors:
        :param layer_sizes:
        """
        self.model_type = model_type
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.reg_layers = reg_layers
        # define input columns
        self.categ_input = None
        self.conti_input = None
        self.cross_input = None
        # record train process
        self.history = []

        self.model = self.get_model()

    def get_model(self, model_type):
        if model_type == 'deep':
            return self.deep_model()
        elif model_type == 'wide':
            return self.wide_model()
        elif model_type == 'deep_wide':
            return self.deep_wide()

    def deep_model(self, category_columns_count, continue_columns):
        categ_inputs = []
        categ_embeds = []

        for i in category_columns_count:
            inputs = Input(shape=(1,), dtype='int32', name=i)
            input_dim = category_columns_count[i]
            output_dim = int(np.ceil(input_dim ** 0.25))
            embedding_layer = Embedding(input_dim=input_dim,
                                        output_dim=output_dim,
                                        input_length=1,
                                        embeddings_regularizer=l2(0),
                                        name=i+"_embedding")(inputs)
            fatten_layer = Flatten()(embedding_layer)
            categ_inputs.append(inputs)
            categ_embeds.append(fatten_layer)

        # input numerical data
        conti_input = Input(shape=(len(continue_columns), ), name='numerical_input')
        conti_dense = Dense(256, activation='relu', use_bias=False, name='numerical_layer')(conti_input)

        # save deep model input layer
        self.categ_input = categ_inputs
        self.conti_input = conti_input

        # concat embedding layer
        concat_embeds = Concatenate(axis=-1, name='concat_embedding')([conti_dense] + categ_embeds)
        bn = BatchNormalization()(concat_embeds)

        # mlp layer
        for i in range(len(self.layers)):
            dense_layer = Dense(self.layers[i],
                                activation='relu',
                                kernel_regularizer=l2(0),
                                name='deep_layer_{}'.format(i))(bn)
            bn = BatchNormalization(name='bn_layer_{}'.format(i))(dense_layer)

        # final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction_layer')(bn)

        # build the deep model
        model = Model(inputs=[categ_inputs, conti_input],
                      outputs=prediction)

        return model

    def wide_model(self, cross_nums):
        # linear model
        wide_input = Input(shape=(cross_nums,), name='wide_input')
        wide_output = Dense(cross_nums, activation='sigmoid', name='wide_output')(wide_input)
        model = Model(inputs=wide_input, outputs=wide_output)
        self.cross_input = wide_input

        return model

    def deep_wide(self, category_columns_count, continue_columns, cross_nums):

        # create deep and wide model
        wide_model = self.wide_model(cross_nums)
        deep_model = self.deep_model(category_columns_count, continue_columns)

        # concat deep and wide input
        cross_input = wide_model.get_layer(name='wide_input').output
        concat_input = [cross_input, self.conti_input] + self.categ_input

        # concat deep and wide output
        deep_output = deep_model.get_layer(name='deep_layer_{}'.format(len(self.layers)-1)).output
        wide_output = cross_input

        a = [deep_output, wide_output]
        concat_output = Concatenate(name='concat_output')([wide_output, deep_output])

        # final prediction layer
        prediction = Dense(1, kernel_initializer='lecun_uniform', name='wd_prediction')(concat_output)

        # build deep and wide model
        model = Model(inputs=concat_input, outputs=prediction)

        return model

    def get_input_data(self, data, numeric_columns, category_columns):
        return data

    def train(self,
              x_train,
              y_train,
              optimizer='adam',
              loss='binary_crossentropy',
              metric='accuracy',
              learning_rate=0.001,
              epoch=1,
              batch_size=32,
              verbose=1,
              validation_split=0.1):

        if optimizer.lower() == 'adagrad':
            op = Adagrad(learning_rate=learning_rate)
        elif self.optimizer.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            op = RMSprop(learning_rate)
        else:
            op = SGD(learning_rate)

        self.model.compile(optimizer=op, loss=loss, metric=metric)
        for i in range(epoch):
            history = self.model.fit(x_train,
                                     y_train,
                                     epochs=1,
                                     batch_size=batch_size,
                                     validation_split=validation_split,
                                     verbose=verbose)

            self.history.append(history)

    def predict(self, x_test):
        # convert
        prediction = self.model.predict(x_test)
        prediction = [1 if x >= 0.5 else 0 for x in prediction]
        return prediction

    def save_model(self, directory):
        """
        save the model
        :param directory:the directory where to put the model
        :return: the absolute path
        """

        # create the dir if not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # remove the model file if exists
        model_name = self.model_type + '.h5'
        model_path = os.path.join(directory, model_name)
        if os.path.exists(model_path):
            os.remove(model_path)

        # save the network graph and weights
        self.model.save(model_path)

        return os.path.realpath(model_path)

    def plot_picture(self):
        return 1


