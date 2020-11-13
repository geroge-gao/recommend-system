import tensorflow as tf
from tensorflow.keras.experimental import WideDeepModel


class Deep_Wide:

    def __init__(self,
                 n_users=10,
                 n_items=10,
                 model_type="NeuMF",
                 user_mlp_embedding_dim=10,
                 item_mlp_embedding_dim=10,
                 user_gmf_embedding_dim=10,
                 item_gmf_embedding_dim=10,
                 n_factors=8,
                 learning_rate=0.1,
                 layers=[20, 10],
                 reg_layers=[0, 0],
                 epochs=1,
                 optimizer='adam',
                 loss='binary_crossentropy',
                 load_pretrain=False,
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
        self.n_users = n_users
        self.n_items = n_items
        self.model_type = model_type
        self.n_factors = n_factors
        self.layers = layers
        self.user_mlp_embedding_dim = user_mlp_embedding_dim
        self.item_mlp_embedding_dim = item_mlp_embedding_dim
        self.user_gmf_embedding_dim = user_gmf_embedding_dim
        self.item_gmf_embedding_dim = item_gmf_embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.load_pretrain = load_pretrain
        self.reg_layers = reg_layers

        self.model = self.get_model()

    def get_model(self):

        return self.get_model()

    def deep_model(self):
        return self.model

    def wide_model(self):
        return self.model

    def deep_wide(self):
        return self.model

    def train(self):
        return self.model

    def predict(self):
        return self.model

    def save_model(self):
        return self.data


