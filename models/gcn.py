import tensorflow as tf
import tensorflow.keras as K
from utils.layers import MGCN, GraphAggregationLayer
from utils.layers import MLP


class DGCN(K.layers.Layer):                         # decoder_adj
    def __init__(self, units, feature_shape, activation=None, dropout_rate=0.5):
        super(DGCN, self).__init__()
        self.edges, self.vertexes, self.nodes = feature_shape
        self.mlp = MLP(units=units, activation=tf.nn.tanh, dropout_rate=dropout_rate)

        self.edges_dense = K.layers.Dense(units=self.edges * self.vertexes * self.vertexes, activation=activation)
        self.edges_dropout = K.layers.Dropout(dropout_rate)

        self.nodes_dense = K.layers.Dense(units=self.vertexes * self.nodes, activation=activation)
        self.nodes_dropout = K.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        output = self.mlp(inputs, training=training)
        edges_logits = tf.reshape(self.edges_dense(output), (-1, self.edges, self.vertexes, self.vertexes))
        edges_logits = tf.transpose((edges_logits + tf.transpose(edges_logits, (0, 1, 3, 2))) / 2, (0, 2, 3, 1))
        edges_logits = self.edges_dropout(edges_logits, training=training)

        nodes_logits = self.nodes_dense(inputs=output)
        nodes_logits = tf.reshape(nodes_logits, (-1, self.vertexes, self.nodes))
        nodes_logits = self.nodes_dropout(nodes_logits, training=training)

        return edges_logits, nodes_logits

    def get_config(self):
        base_config = super(DGCN, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EGCN(K.layers.Layer):         # encoder_rgcn

    def __init__(self, units, edge_feature, activation=None, dropout_rate=0.5):
        super(EGCN, self).__init__()
        graph_convolution_units, auxiliary_units = units    # (128, 64) 128
        self.edge_feature = edge_feature
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.mgcn = MGCN(graph_convolution_units, self.edge_feature,
                         dropout_rate=self.dropout_rate,
                         activation=self.activation)
        self.read_out = GraphAggregationLayer(units=auxiliary_units, activation=tf.nn.tanh,
                                              dropout_rate=self.dropout_rate)

    def call(self, inputs, training=False):
        output = self.mgcn(inputs, training=training)
        _, hidden_tensor, node_tensor = inputs
        annotations = tf.concat(
            (output, hidden_tensor, node_tensor) if hidden_tensor is not None else (output, node_tensor), -1)
        output = self.read_out(annotations, training=training)
        return output

    def get_config(self):
        base_config = super(DGCN, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
