import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from models import postprocess_logits
from utils.layers import multi_dense_layers
from utils.utils import reward
from utils.sparse_molecular_dataset import SparseMolecularDataset

data = SparseMolecularDataset()


class GraphGANModel(keras.layers.Layer):

    def __init__(self, vertexes, edges, nodes, embedding_dim, decoder_units, discriminator_units,
                 decoder, discriminator, soft_gumbel_softmax=False, hard_gumbel_softmax=False,
                 batch_discriminator=True, batch=8, training=False):
        """

        :param vertexes: the atoms num of molecular
        :param edges: the bond num
        :param nodes: the atom num type
        :param embedding_dim: embedding dim
        :param decoder_units:
        :param discriminator_units:
        :param decoder: the generate part: embedding to edge_logits and node_logits
        :param discriminator: the discriminator part:
        :param soft_gumbel_softmax:
        :param hard_gumbel_softmax:
        :param batch_discriminator:
        """
        super(GraphGANModel, self).__init__()
        self.vertexes, self.edges, self.nodes = vertexes, edges, nodes
        self.embedding_dim, self.decoder_units = embedding_dim, decoder_units
        self.discriminator_units = discriminator_units
        self.batch_discriminator = batch_discriminator

        self.dropout_rate = 0.
        self.soft_gumbel_softmax = tf.constant(
            soft_gumbel_softmax)  # tf.placeholder_with_default(soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.constant(
            hard_gumbel_softmax)  # tf.placeholder_with_default(hard_gumbel_softmax, shape=())
        self.temperature = 1.  # tf.placeholder_with_default(1., shape=())
        self.batch = batch
        self.training = training

        # self.decoder = DGCN()
        # self.discriminator = EGCN()
        self.decoder, self.discriminator = decoder, discriminator

        # ###

        # self.edges_labels = tf.placeholder(dtype=tf.int64, shape=(None, vertexes, vertexes))
        # self.nodes_labels = tf.placeholder(dtype=tf.int64, shape=(None, vertexes))
        # self.embeddings_dim = (None, embedding_dim)     #tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim))

        # self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        # self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        # self.adjacency_tensor = tf.one_hot(self.edges_labels, depth=edges, dtype=tf.float32)
        # self.node_tensor = tf.one_hot(self.nodes_labels, depth=nodes, dtype=tf.float32)

    def call(self, inputs):
        data_a, data_x, data= inputs
        self.embeddings = self.sample_z(self.batch)

        # generate
        self.edges_logits, self.nodes_logits = self.decoder(self.embeddings, self.decoder_units, self.vertexes,
                                                            self.edges, self.nodes, training=False,
                                                            dropout_rate=self.dropout_rate)
        (self.edges_softmax, self.nodes_softmax), \
        (self.edges_argmax, self.nodes_argmax), \
        (self.edges_gumbel_logits, self.nodes_gumbel_logits), \
        (self.edges_gumbel_softmax, self.nodes_gumbel_softmax), \
        (self.edges_gumbel_argmax, self.nodes_gumbel_argmax) = postprocess_logits(
            (self.edges_logits, self.nodes_logits), temperature=self.temperature)

        self.adjacency_tensor = tf.one_hot(data_a, depth=self.edges, dtype=tf.float32)
        self.node_tensor = tf.one_hot(data_x, depth=self.nodes, dtype=tf.float32)

        # self.rewardR = reward(mols)  # 采样分子

        n, e = self.nodes_gumbel_argmax, self.edges_gumbel_argmax
        n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
        gen_mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
        self.rewardF = reward(gen_mols)  # 生成分子

        self.edges_hat = tf.case([(self.soft_gumbel_softmax, lambda: self.edges_gumbel_softmax),  #
                                  (self.hard_gumbel_softmax, lambda: tf.stop_gradient(
                                      self.edges_gumbel_argmax - self.edges_gumbel_softmax) + self.edges_gumbel_softmax)],
                                 #
                                 default=lambda: self.edges_softmax,
                                 exclusive=True)

        self.nodes_hat = tf.case([(self.soft_gumbel_softmax, lambda: self.nodes_gumbel_softmax),
                                  (self.hard_gumbel_softmax, lambda: tf.stop_gradient(
                                      self.nodes_gumbel_argmax - self.nodes_gumbel_softmax) + self.nodes_gumbel_softmax)],
                                 default=lambda: self.nodes_softmax,
                                 exclusive=True)

        self.logits_real, self.features_real = self.D_x((self.adjacency_tensor, None, self.node_tensor),
                                                        units=self.discriminator_units)

        self.logits_fake, self.features_fake = self.D_x((self.edges_hat, None, self.nodes_hat),
                                                        units=self.discriminator_units)

        self.value_logits_real = self.V_x((self.adjacency_tensor, None, self.node_tensor),
                                          units=self.discriminator_units)

        self.value_logits_fake = self.V_x((self.edges_hat, None, self.nodes_hat), units=self.discriminator_units)

    def D_x(self, inputs, units):
        # with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

        # if type(inputs) is tuple:
        #     for x in inputs:
        #         print("inputs hape: ", x.shape)
        outputs0 = self.discriminator(inputs, units=units[:-1], training=self.training,
                                      dropout_rate=self.dropout_rate)

        outputs1 = multi_dense_layers(outputs0, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                      dropout_rate=self.dropout_rate)

        if self.batch_discriminator:
            outputs_batch = keras.layers.Dense(units[-2] // 8, activation=tf.tanh)(outputs0)
            outputs_batch = keras.layers.Dense(units[-2] // 8,
                                               activation=tf.nn.tanh)(
                tf.reduce_mean(outputs_batch, 0, keepdims=True), )
            outputs_batch = tf.tile(outputs_batch, (tf.shape(outputs0)[0], 1))

            outputs1 = tf.concat((outputs1, outputs_batch), -1)

        outputs = keras.layers.Dense(units=1)(outputs1)

        return outputs, outputs1

    def V_x(self, inputs, units):
        # with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
        outputs = self.discriminator(inputs, units=units[:-1], training=self.training,
                                     dropout_rate=self.dropout_rate)

        outputs = multi_dense_layers(outputs, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                     dropout_rate=self.dropout_rate)

        outputs = keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(outputs)

        return outputs

    def sample_z(self, batch):
        return np.random.normal(0, 1, size=(batch, self.embedding_dim))
