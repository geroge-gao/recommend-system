from utils.layer_utils import FM, Reduce
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Add, Flatten, RepeatVector
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.initializers import GlorotNormal, RandomNormal

class deepFM():

    """

    """

    def __init__(self,
                 dense_columns,
                 sparse_columns,
                 model_type='deepfm',
                 layers=[512, 256, 128],
                 reg_layers=[0, 0, 0],
                 embed_output_dim=4,
                 seed=2020):
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
        self.callbacks = None
        self.seed = seed
        self.model = self.deepfm()


    def deepfm(self):

        # define dense layer
        dense_input = Input(len(self.dense_columns), name='dense_input', dtype='float32')

        # define embedding layer
        sparse_inputs = []
        fm_embeds = []
        linear_embeds = []
        for k, v in self.sparse_columns:
            sparse_input = Input(1, name=k, dtype='int32')
            sparse_inputs.append(sparse_input)
            input_dim = v,
            output_dim = self.output_dim
            fm_embed = Embedding(input_dim=input_dim,
                                 output_dim=output_dim,
                                 embeddings_initializer=tf.keras.initializers.RandomNormal,
                                 embeddings_regularizer=l2(0),
                                 )(sparse_input)

            linear_embed = Embedding(input_dim=input_dim,
                                     output_dim=1,
                                     embeddings_initializer=tf.keras.initializers.RandomNormal,
                                     embeddings_regularizer=l2(0),
                                     )(sparse_input) # [None, input_dim ,1]
            # get embedding layer
            linear_embeds.append(linear_embed)
            fm_embeds.append(fm_embed)

        # linear part
        linear_dense = Dense(1)(dense_input)
        linear_concat = Concatenate(axis=-1)(linear_embeds)
        linear_embed = Reduce(axis=1)(linear_concat)
        linear_vector = Add([linear_dense, linear_embed])

        # FM part
        fm_dense = RepeatVector(1)(Dense(self.output_dim)(dense_input))  # [batch_size, 1, output_dim]
        fm_input = Concatenate(axis=-1)([fm_dense] + fm_embeds)
        fm_vector = FM()(fm_input)  # [batch_size, 1]

        # DNN part
        dnn_vector = Flatten()(fm_input)

        for i in range(len(self.layers)):
            dnn_vector = Dense(self.layers[i],
                               kernel_initializer=GlorotNormal(seed=self.seed),
                               kernel_regularizer=l2(self.reg_layers[0]),
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



