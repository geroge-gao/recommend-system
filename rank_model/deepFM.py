from utils.layer_utils import FM
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Add, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class deepFM():

    """

    """

    def __init__(self,
                 dense_columns,
                 sparse_columns,
                 model_type='deepfm',
                 layers=[512, 256, 128],
                 reg_layers=[0, 0, 0],
                 embed_output_dim=4,):
        """
        :param dense_columns: a dict, (column_name, column_unique_count)
        :param sparse_columns: a list of sparse column
        :param model_type: model type, such as 'fm' and 'deepfm'
        :param layers: DNN layer size
        :param reg_layers: Regularization parameters of neural networks
        :param embed_output_dim: embedding output dim
        """

        self.layers = layers
        self.reg_layers = reg_layers
        self.output_dim = embed_output_dim
        self.dense_columns = dense_columns
        self.sparse_columns = sparse_columns
        self.model = self.deepfm()


    def deepfm(self):

        # input numeric data
        dense_input = Input(len(self.dense_columns), name=+'numeric_input', dtype='int32')

        # input category data
        sparse_inputs = []
        for k, v in self.sparse_columns:
            sparse_input = Input(1, name=k, dtype='float32')
            emb_input_dim = v
            emb_output_dim = self.output_dim
            embedding = Embedding(input_dim=emb_input_dim,
                                  output_dim=emb_output_dim,
                                  input_length=1,
                                  embeddings_regularizer=l2(0),
                                  name=k + "_embedding"
                                  )(sparse_input)

            sparse_inputs.append(embedding)

        # linear input
        dense_linear = Dense(1)(dense_input)
        sparse_linear = Concatenate(axis=-1)(sparse_inputs)
        sparse_linear = tf.reduce_sum(sparse_linear, axis=1)




        dense_input
        linear_input = Concatenate(axis=-1)([dense_input] + sparse_inputs)
        fm_input = Concatenate(axis=-1)(sparse_inputs)
        dnn_input = linear_input

        # linear part
        linear_output = Dense(1, activation='linear', use_bias=True, name='linear')(linear_input)

        # fm part
        fm_output = FM()(fm_input)

        # dnn part
        dense_layer = Flatten()(dnn_input)

        for i in range(self.layers):
            dense_layer = Dense(self.layers[i],
                                activation='relu',
                                kernel_initializer='random_uniform',
                                name='mlp_{}_layer'.format(i))(dense_layer)



        # prediction layer

        model = Model()
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

        print('result', self.model.summary())

        early_stop = EarlyStopping(monitor='val_loss', patience=3)

        self.callbacks = self.model.fit(x_train,
                                        y_train,
                                        epochs=n_epoch,
                                        batch_size=n_batch_size,
                                        verbose=n_verbose,
                                        validation_split=validation_ratio,
                                        callbacks=[early_stop])

