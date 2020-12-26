from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_normal, zeros


class FM(Layer):
    """

        input shape (batch_size, field_dim ,embedding_dim)
    """

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        # Creates the variables of the layer
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # This is where the layer's logic lives.

        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embedding_input = inputs
        print('input_shape', inputs.shape)

        # square_of_sum's shape: [batch_size, 1 ,embedding_dim]
        square_of_sum = tf.square(tf.reduce_sum(embedding_input, axis=1, keepdims=True))
        # sum_of_square's shape: [batch_size, 1,embedding_dim]
        sum_of_square = tf.reduce_sum(tf.square(embedding_input), axis=1, keepdims=True)
        # cross term's shape: [batch_size, 1]
        cross_term = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=-1, keepdims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        # Computes the output shape of the layer.
        return (None, 1)


class ReduceLayer(Layer):
    """
    Reduce the dim of data
    """
    def __init__(self, axis=-1,  **kwargs):
        super(ReduceLayer, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape, input_mask=None):

        super(ReduceLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            if K.ndim(inputs) != K.ndim(mask):
                mask = K.repeat(mask, inputs.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask
        sparse_input = tf.reduce_sum(inputs, axis=self.axis, keepdims=False)
        return sparse_input

    def compute_output_shape(self, input_shape):
        return (None, 1)
    

class CrossLayer(Layer):
    """
    Cross Layer for deep cross network
    """
    def __init__(self, layer_num, layer_reg, **kwargs):
        self.layer_num = layer_num
        self.layer_reg = layer_reg
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = []
        self.b = []
        for i in range(self.layer_num):
            self.W.append(self.add_weight(name='kernel',
                                          shape=(int(input_shape[-1]), 1),
                                          initializer=glorot_normal(),
                                          regularizer=l2(0),
                                          trainable=True))
            self.b.append(self.add_weight(name='bias',
                                          shape=(int(input_shape[-1]), 1),
                                          initializer=zeros(),
                                          trainable=True))

        super(CrossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        # add a dim
        x0 = K.reshape(inputs, (-1, inputs.shape[1], 1))
        xl = K.transpose(x0, [0, 2, 1])  # [batch_size, 1, embedding_dim]

        for i in range(self.layer_num):
            cross_out = 1



        return 1

    def compute_output_shape(self, input_shape):
        return input_shape





