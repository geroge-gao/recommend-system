from tensorflow.keras.layers import Layer, Dense, Dropout, Input
from tensorflow.keras import Model, activations

from tensorflow import keras as K


class FM(Layer):
    def __init__(self, output_dim, latent=10,  activation='relu', **kwargs):
        self.latent = latent
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        # 创建改神经层的变量值,确认输入数据的shape
        self.b = self.add_weight(name='W0',
                                 shape=(self.output_dim,),
                                 trainable=True,
                                 initializer='zeros')
        self.w = self.add_weight(name='W',
                                 shape=(input_shape[1], self.output_dim),
                                 trainable=True,
                                 initializer='random_uniform')
        self.v = self.add_weight(name='V',
                                 shape=(input_shape[1], self.latent),
                                 trainable=True,
                                 initializer='random_uniform')
        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # This is where the layer's logic lives.
        x = inputs
        x_square = K.square(x)
        xv = K.square(K.dot(x, self.v))
        xw = K.dot(x, self.w)

        p = 0.5*K.sum(xv-K.dot(x_square, K.square(self.v)), 1)
        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)
        f = xw + rp + self.b
        output = K.reshape(f, (-1, self.output_dim))

        return output

    def compute_output_shape(self, input_shape):
        # 计算输出的数据类型
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim
