import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Input,
                                     Dense,
                                     Embedding,
                                     Flatten,
                                     Concatenate,
                                     BatchNormalization)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, RMSprop


class Deep_Wide:

    def __init__(self,
                 model_type="deep",
                 layers=[512, 256, 128],
                 reg_layers=[0, 0, 0],
                 category_columns_count=None,
                 continous_nums=0,
                 cross_nums=0
                 ):
        """
        wide and deep model
        :param n_users: numbers of user in the dataset
        :param n_items:
        :param model_type:
        :param n_factors:
        :param layer_sizes:
        """
        self.model_type = model_type
        self.layers = layers
        self.reg_layers = reg_layers
        # define input columns
        self.categ_input = None
        self.conti_input = None
        self.cross_input = None
        # column information
        self.category_columns_count = category_columns_count
        self.continous_nums = continous_nums
        self.cross_nums = cross_nums

        # record train process
        self.callbacks = None

        self.model = self.get_model(model_type)

    def get_model(self, model_type):
        if model_type == 'deep':
            return self.deep_model()
        elif model_type == 'wide':
            return self.wide_model()
        elif model_type == 'deep_wide':
            return self.deep_wide()

    def deep_model(self):
        categ_inputs = []
        categ_embeds = []

        for i in self.category_columns_count:
            inputs = Input(shape=(1,), dtype='int32', name=i)
            input_dim = self.category_columns_count[i]
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
        conti_input = Input(shape=(self.continous_nums, ), name='continous_input')
        conti_dense = Dense(256, activation='relu', use_bias=False, name='continous_layer')(conti_input)

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

    def wide_model(self):
        # linear model
        wide_input = Input(shape=(self.cross_nums,), name='wide_input')
        wide_output = Dense(1, activation='sigmoid', name='wide_output')(wide_input)
        model = Model(inputs=wide_input, outputs=wide_output)
        self.cross_input = wide_input

        return model

    def deep_wide(self):

        # create deep and wide model
        wide_model = self.wide_model()
        deep_model = self.deep_model()

        # concat deep and wide input
        cross_input = wide_model.get_layer(name='wide_input').output
        concat_input = [cross_input, self.conti_input] + self.categ_input

        # concat deep and wide output
        deep_output = deep_model.get_layer(name='deep_layer_{}'.format(len(self.layers)-1)).output
        wide_output = cross_input

        concat_output = Concatenate(name='concat_output')([wide_output, deep_output])

        # final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='wd_prediction')(concat_output)

        # build deep and wide model
        model = Model(inputs=concat_input, outputs=prediction)

        return model

    def train(self,
              x_train,
              y_train,
              opt='adam',
              loss_='binary_crossentropy',
              metric_='accuracy',
              lr=0.001,
              n_epoch=1,
              n_batch_size=32,
              n_verbose=1,
              validation_ratio=0.1,
              pretrain=False,
              wide_dir='',
              deep_dir=''):

        if opt.lower() == 'adagrad':
            op = Adagrad(learning_rate=lr)
        elif opt.lower() == 'adam':
            op = Adam(learning_rate=lr)
        elif opt.lower() == 'rmsprop':
            op = RMSprop(lr)
        else:
            op = SGD(lr)

        if pretrain:
            self.load_pretrain_model(wide_dir, deep_dir)
        print(metric_ == 'accuracy', metric_)
        self.model.compile(optimizer=op, loss=loss_, metrics=[metric_])

        early_stop = EarlyStopping(monitor='val_loss', patience=3)

        self.callbacks = self.model.fit(x_train,
                                        y_train,
                                        epochs=n_epoch,
                                        batch_size=n_batch_size,
                                        verbose=n_verbose,
                                        validation_split=validation_ratio,
                                        callbacks=[early_stop])

    def evaluate_model(self, data, label):
        loss, acc = self.model.evaluate(data, label)
        print('test loss: {}, test acc: {}'.format(loss, acc))

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
        # visualize training results
        # 绘制训练 & 验证的准确率值
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.callbacks.history['acc'])
        plt.plot(self.callbacks.history['val_acc'])
        plt.title(self.model_type + ' accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower right')

        # 绘制训练 & 验证的损失值
        plt.subplot(1, 2, 2)
        plt.plot(self.callbacks.history['loss'])
        plt.plot(self.callbacks.history['val_loss'])
        plt.title(self.model_type + ' loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower right')
        plt.show()





