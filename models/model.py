import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras as K
from tensorflow.keras.models import Model
from models import postprocess_logits
from utils.layers import multi_dense_layers
from utils.utils import reward
from utils.sparse_molecular_dataset import SparseMolecularDataset
from models import encoder_rgcn
from models.gcn import DGCN, EGCN
from utils.layers import MLP


class MolGan(Model):

    def __init__(self, vertexes, edges, nodes, embedding_dim, decoder_units, discriminator_units,
                 soft_gumbel_softmax=False, hard_gumbel_softmax=False,
                 batch_discriminator=True, batch=8, training=True, learning_rate=0.001, dropout_rate=0.5,
                 temperature=1.):
        super(MolGan, self).__init__()
        self.vertexes, self.edges, self.nodes = vertexes, edges, nodes
        self.embedding_dim, self.decoder_units = embedding_dim, decoder_units
        self.discriminator_units = discriminator_units
        self.batch_discriminator = batch_discriminator

        self.learning_rate = learning_rate

        self.dropout_rate = dropout_rate
        self.soft_gumbel_softmax = tf.constant(
            soft_gumbel_softmax)  # tf.placeholder_with_default(soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.constant(
            hard_gumbel_softmax)  # tf.placeholder_with_default(hard_gumbel_softmax, shape=())
        self.temperature = temperature  # tf.placeholder_with_default(1., shape=())
        self.batch = batch
        self.training = training
        # encoder

        # DGCN/ decoder_adj
        self.G_x = DGCN(units=self.decoder_units, feature_shape=(edges, vertexes, nodes), activation=None,
                        dropout_rate=self.dropout_rate)

        self.D_x = Discriminate(self.discriminator_units, edge_feature=self.edges, dropout_rate=self.dropout_rate)
        self.V_x = Discriminate(self.discriminator_units, edge_feature=self.edges, batch_discriminator=False,
                                dropout_rate=self.dropout_rate)

    def call(self, inputs=None, training=False):

        if inputs is not None:
            assert len(inputs) == 2, "inputs must contain adjacent and node feature matrix."
            data_a, data_x = inputs

        # decoder_units, vertexes, edges, nodes = inputs
        sample_encoding = self.sample_z(self.batch)
        edges_logits, nodes_logits = self.G_x(sample_encoding)

        (edges_softmax, nodes_softmax), \
        (edges_argmax, nodes_argmax), \
        (edges_gumbel_logits, nodes_gumbel_logits), \
        (edges_gumbel_softmax, nodes_gumbel_softmax), \
        (edges_gumbel_argmax, nodes_gumbel_argmax) = postprocess_logits((edges_logits, nodes_logits),
                                                                        temperature=self.temperature)

        edges_hat = tf.case([(self.soft_gumbel_softmax, lambda: edges_gumbel_softmax),
                             (self.hard_gumbel_softmax, lambda: tf.stop_gradient(
                                 edges_gumbel_argmax - edges_gumbel_softmax) + edges_gumbel_softmax)],
                            default=lambda: edges_softmax,
                            exclusive=True)

        nodes_hat = tf.case([(self.soft_gumbel_softmax, lambda: nodes_gumbel_softmax),
                             (self.hard_gumbel_softmax, lambda: tf.stop_gradient(
                                 nodes_gumbel_argmax - nodes_gumbel_softmax) + nodes_gumbel_softmax)],
                            default=lambda: nodes_softmax,
                            exclusive=True)

        if not training:
            return nodes_gumbel_argmax, edges_gumbel_argmax

        # sample real
        if inputs is None:
            raise ValueError
        adjacency_tensor = tf.one_hot(data_a, depth=self.edges, dtype=tf.float32)
        node_tensor = tf.one_hot(data_x, depth=self.nodes, dtype=tf.float32)

        logits_real, features_real = self.D_x((adjacency_tensor, None, node_tensor), training=training)
        value_logits_real, _ = self.V_x((adjacency_tensor, None, node_tensor), training=training)

        # generate
        logits_fake, features_fake = self.D_x((edges_hat, None, nodes_hat), training=training)
        value_logits_fake, _ = self.V_x((edges_hat, None, nodes_hat), training=training)

        # print("D_x: ", self.D_x.trainable_variables)
        real_sample = (adjacency_tensor, node_tensor, logits_real, features_real, value_logits_real)
        gen_sample = (edges_hat, nodes_hat, logits_fake, features_fake, value_logits_fake)
        gen_raw = (edges_softmax, nodes_softmax, edges_argmax, nodes_argmax, edges_gumbel_logits, nodes_gumbel_logits,
                   edges_gumbel_softmax, nodes_gumbel_softmax, edges_gumbel_argmax, nodes_gumbel_argmax)
        return real_sample, gen_sample, gen_raw

    def sample_z(self, batch):
        return np.random.normal(0, 1, size=(batch, self.embedding_dim))

    def get_config(self):
        base_config = super(MolGan, self).get_config()
        base_config['output_dim'] = self.atom_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Discriminate(keras.layers.Layer):

    def __init__(self, units, edge_feature, batch_discriminator=True, dropout_rate=0.3):
        super(Discriminate, self).__init__()
        self.units = units
        self.batch_discriminator = batch_discriminator
        self.dropout_rate = dropout_rate
        self.edge_feature = edge_feature

        self.egcn = EGCN(units=self.units[:-1], edge_feature=self.edge_feature, activation=None,
                         dropout_rate=self.dropout_rate)
        self.mlp = MLP(units=units[-1], activation=tf.tanh, dropout_rate=self.dropout_rate)
        self.out_dense = K.layers.Dense(units=1)
        if self.batch_discriminator:
            self.batch_discriminator_dense = [K.layers.Dense(units[-2] // 8, activation=tf.tanh),
                                              K.layers.Dense(units[-2] // 8, activation=tf.tanh)]

    def call(self, inputs, training=False):

        discriminator_out = self.egcn(inputs=inputs, training=training)
        multi_dense_out = self.mlp(inputs=discriminator_out)

        if self.batch_discriminator:
            outputs_batch = self.batch_discriminator_dense[0](discriminator_out)
            outputs_batch = tf.reduce_mean(outputs_batch, 0, keepdims=True)
            outputs_batch = self.batch_discriminator_dense[1](outputs_batch)
            outputs_batch = tf.tile(outputs_batch, (tf.shape(discriminator_out)[0], 1))
            multi_dense_out = tf.concat((multi_dense_out, outputs_batch), -1)

        discriminator_score = self.out_dense(multi_dense_out)
        return discriminator_score, multi_dense_out


# def multi_dense_layers(inputs, units, training, activation=None, dropout_rate=0.):
#     hidden_tensor = inputs  # units (128, 64), 128
#     for u in units:
#         hidden_tensor = K.layers.Dense(u, activation=activation)(hidden_tensor)
#         hidden_tensor = K.layers.Dropout(dropout_rate)(hidden_tensor, training=training)
#
#     return hidden_tensor


# class Generate(keras.layers.Layer):
#     def __init__(self, batch=8, training=True, dropout_rate=0.001, embedding_dim=128, inputs_shape=None):
#         super(Generate, self).__init__()
#         self.batch = batch
#         self.training = training
#         self.dropout_rate = dropout_rate
#         self.embedding_dim = embedding_dim
#
#         self.inputs_shape = inputs_shape
#
#         self.multi_dense = [K.layers.Dense(u, tf.nn.relu) for u in (128, 256, 512)]
#         self.multi_drop = [K.layers.Dropout(dropout_rate) for _ in (1, 2, 3)]
#
#     def call(self, inputs=None):
#         decoder_units, vertexes, edges, nodes = self.inputs_shape
#         z = self.sample_z(self.batch)
#         output = z.copy()
#         for i in range(len(self.multi_dense)):
#             output = self.multi_dense[i](output)
#             output = self.multi_drop[i](output)
#         # output = multi_dense_layers(z, decoder_units, activation=tf.nn.tanh, dropout_rate=self.dropout_rate,
#         #                             training=self.training)
#
#         # with tf.variable_scope('edges_logits'):
#         edges_logits = tf.reshape(keras.layers.Dense(units=edges * vertexes * vertexes,
#                                                      activation=None)(output), (-1, edges, vertexes, vertexes))
#
#         edges_logits = tf.transpose((edges_logits + tf.transpose(edges_logits, (0, 1, 3, 2))) / 2, (0, 2, 3, 1))
#
#         edges_logits = keras.layers.Dropout(self.dropout_rate)(edges_logits, training=self.training)
#
#         # with tf.variable_scope('nodes_logits'):
#         nodes_logits = keras.layers.Dense(units=vertexes * nodes, activation=None)(inputs=output)
#         nodes_logits = tf.reshape(nodes_logits, (-1, vertexes, nodes))
#         nodes_logits = keras.layers.Dropout(self.dropout_rate)(nodes_logits, training=self.training)
#         return edges_logits, nodes_logits
#
#     def sample_z(self, batch):
#         return np.random.normal(0, 1, size=(batch, self.embedding_dim))
