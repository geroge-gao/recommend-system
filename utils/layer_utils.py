from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.regularizers import l2


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

        if K.ndim(inputs) == 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embedding_input = inputs

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


class Reduce(Layer):
    """
    Linear layer for deepFM
    """
    def __init__(self, axis=-1,  **kwargs):
        self.axis = axis
        super(Reduce, **kwargs)

    def build(self, input_shape):

        super(Reduce, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sparse_input = tf.reduce_sum(inputs, axis=self.axis, keepdims=False)
        return sparse_input

    def compute_output_shape(self, input_shape):
        return (None, 1)



