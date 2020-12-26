import os
import matplotlib.pyplot as plt
from utils.layer_utils import FM, ReduceLayer
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Add, Flatten, RepeatVector
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf
# from tensorflow.keras.initializers import GlorotNormal


class deepFM():

    """

    """

    def __init__(self,
                 dense_columns,
                 sparse_columns,
                 layer_size=[512, 256, 128],
                 layers_reg=[0, 0, 0],
                 embed_dim=4,
                 seed=2020):
        """
        :param dense_columns: a dict, (column_name, column_unique_count)
        :param sparse_columns: a list of sparse column
        :param model_type: model type, such as 'fm' and 'deepfm'
        :param layers: DNN layer size
        :param reg_layers: Regularization parameters of neural networks
        :param embed_output_dim: embedding output dim
        """

        self.layers = layer_size
        self.layers_reg = layers_reg
        self.embed_dim = embed_dim
        self.dense_columns = dense_columns
        self.sparse_columns = sparse_columns
        self.callbacks = None
        self.seed = seed
        self.model = self.deepfm()


    def deepfm(self):

        # define dense layer
        sparse_inputs = []
        dense_inputs = []
        # for c in self.dense_columns:
        #     dense_input = Input(1, name=c+'_input', dtype='float32')
        #     dense_inputs.append(dense_input)

        dense_input = Input(len(self.dense_columns), name='dense_input', dtype='float32')

        # define embedding layer

        fm_embeds = []
        linear_embeds = []
        for k, v in self.sparse_columns.items():
            sparse_input = Input(1, name=k+'_input', dtype='int32')
            sparse_inputs.append(sparse_input)
            embed_input_dim = v
            embed_output_dim = self.embed_dim

            fm_embed = Embedding(input_dim=embed_input_dim,
                                 output_dim=embed_output_dim,
                                 mask_zero=True)(sparse_input)

            linear_embed = Embedding(input_dim=embed_input_dim,
                                     output_dim=1,
                                     mask_zero=True)(sparse_input)  # [None, input_dim ,1]

            linear_embeds.append(linear_embed)
            fm_embeds.append(fm_embed)

        # linear part
        # dense_concat = Concatenate(axis=-1)(dense_inputs)
        linear_dense = Dense(1)(dense_input)
        linear_concat = Concatenate(axis=1)(linear_embeds)
        linear_flatten = ReduceLayer(axis=1)(linear_concat)
        linear_vector = Add()([linear_dense, linear_flatten])  # [batch_size, 1]

        # FM part
        fm_dense = RepeatVector(1)(Dense(self.embed_dim)(dense_input))  # [batch_size, 1, output_dim]
        fm_input = Concatenate(axis=-1)([fm_dense] + fm_embeds)
        fm_vector = FM()(fm_input)  # [batch_size, 1]

        # DNN part
        dnn_vector = Flatten()(fm_input)

        for i in range(len(self.layers)):
            dnn_vector = Dense(self.layers[i],
                               kernel_initializer=tf.keras.initializers.RandomNormal,
                               kernel_regularizer=l2(self.layers_reg[i]),
                               use_bias=True,
                               activation='relu',
                               name='dnn_layer_{}'.format(i))(dnn_vector)

        # define deepfm model input
        final_input = Concatenate(axis=-1)([linear_vector, fm_vector, dnn_vector])

        # define final prediction layer
        prediction = Dense(1, activation='sigmoid')(final_input)

        # build deepFM model
        model = Model(inputs=[dense_input] + sparse_inputs, outputs=prediction)

        return model

    def train(self, x_train, y_train, opt='adam', model_loss='binary_crossentropy', model_metric='accuracy', learning=0.001,
              n_epoch=3, n_batch_size=32, n_verbose=1, validation_ratio=0.2, pretrain=False, model_dir=''):

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



