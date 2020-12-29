import os
import matplotlib.pyplot as plt
from utils.layer_utils import ReduceLayer, CrossLayer
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Add, Flatten, RepeatVector
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform
import tensorflow as tf

class DCN:

    """
    deep cross network
    """

    def __init__(self,
                 dense_columns,
                 sparse_columns,
                 embed_dim=4,
                 cross_layer_nums=2,
                 dnn_layers=[32, 64, 128],
                 layer_reg=[0, 0 , 0],
                 seed=2020):

        self.dense_columns = dense_columns
        self.sparse_columns = sparse_columns
        self.embed_dim = embed_dim
        self.cross_layer_nums = cross_layer_nums
        self.layers = dnn_layers
        self.layer_reg = layer_reg
        self.seed = seed
        self.callbacks = None
        self.model = self.deep_cross_model()

    def deep_cross_model(self):

        # define dense input
        dense_input = Input(len(self.dense_columns), name='dense_input', dtype='float32')

        # define sparse input
        sparse_inputs = []
        sparse_embeds = []
        for k, v in self.sparse_columns.items():
            sparse_input = Input(1, name=str(k) + '_input', dtype='int32')
            sparse_inputs.append(sparse_input)
            embed_input_dim = v
            embed_output_dim = self.embed_dim

            sparse_embed = Embedding(input_dim=embed_input_dim,
                                     output_dim=embed_output_dim,
                                     name=k + 'embedding',
                                     embeddings_initializer=RandomUniform(seed=self.seed)
                                     )(sparse_input)

            sparse_embeds.append(sparse_embed)

        # sparse_embed_input = Concatenate()(sparse_embeds)
        sparse_embed_flatten = Flatten()(sparse_embeds[0])  #
        dnn_input = Concatenate()([dense_input, sparse_embed_flatten])

        # crossNet part
        cross_layer = CrossLayer(self.cross_layer_nums)(dnn_input)

        # dnn part
        dnn_layer = Dense(self.layers[0], activation='relu', name='mlp_layer_0')(dnn_input)
        for i in range(1, len(self.layers)):
            dnn_layer = Dense(self.layers[i],
                              activation='relu',
                              kernel_regularizer=l2(self.layers[i]),
                              name='mlp_layer_{}'.format(i))(dnn_layer)

        # combine the output of dnn and cross net
        combine_input = Concatenate()([cross_layer, dnn_layer])

        # prediction
        prediction = Dense(1, activation='sigmoid', name='prediction')(combine_input)

        # build the model
        model = Model(inputs=[dense_input] + sparse_inputs, outputs=prediction)

        return model

    def train(self, x_train, y_train, opt='adam', model_loss='binary_crossentropy', model_metric='accuracy',
              learning=0.001, n_epoch=3, batchsize=32, n_verbose=1, validation_ratio=0.2, pretrain=False,
              model_dir=''):

        if opt.lower() == 'adagrad':
            op = Adagrad(learning_rate=learning)
        elif opt.lower() == 'adam':
            op = Adam(learning_rate=learning)
        elif opt.lower() == 'rmsprop':
            op = RMSprop(learning)
        else:
            op = SGD(learning)

        if pretrain:
            self.load_pretrain_model(model_dir)

        self.model.compile(optimizer=op, loss=model_loss, metrics=[model_metric])
        early_stop = EarlyStopping(monitor='val_loss', patience=3)

        self.callbacks = self.model.fit(x_train,
                                        y_train,
                                        epochs=n_epoch,
                                        batch_size=batchsize,
                                        verbose=n_verbose,
                                        validation_split=validation_ratio,
                                        callbacks=[early_stop])

    def evaluate_model(self, data, label):
        loss, acc = self.model.evaluate(data, label)
        print('test loss: {}, test acc: {}'.format(loss, acc))

    def predict(self, x_test):
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


