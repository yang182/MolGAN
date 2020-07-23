import tensorflow as tf
from tensorflow import keras as K


def graph_convolution_layer(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    adj = tf.transpose(adjacency_tensor[:, :, :, 1:], (0, 3, 1, 2))

    annotations = tf.concat((hidden_tensor, node_tensor), -1) if hidden_tensor is not None else node_tensor

    output = tf.stack([K.layers.Dense(units=units)(inputs=annotations) for _ in range(adj.shape[1])], 1)

    output = tf.matmul(adj, output)
    output = tf.reduce_sum(output, 1) + K.layers.Dense(units=units)(inputs=annotations)
    output = activation(output) if activation is not None else output
    output = K.layers.Dropout(dropout_rate)(output, training=training)

    return output


def graph_aggregation_layer(inputs, units, training, activation=None, dropout_rate=0.):
    i = K.layers.Dense(units=units, activation=tf.nn.sigmoid)(inputs)
    j = K.layers.Dense(units=units, activation=activation)(inputs)
    output = tf.reduce_sum(i * j, 1)
    output = activation(output) if activation is not None else output
    output = K.layers.Dropout(dropout_rate)(output, training=training)

    return output


def multi_dense_layers(inputs, units, training, activation=None, dropout_rate=0.):
    hidden_tensor = inputs  # units (128, 64), 128
    for u in units:
        hidden_tensor = K.layers.Dense(u, activation=activation)(hidden_tensor)
        hidden_tensor = K.layers.Dropout(dropout_rate)(hidden_tensor, training=training)

    return hidden_tensor


def multi_graph_convolution_layers(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    for u in units:
        hidden_tensor = graph_convolution_layer(inputs=(adjacency_tensor, hidden_tensor, node_tensor),
                                                units=u, activation=activation, dropout_rate=dropout_rate,
                                                training=training)

    return hidden_tensor


class GCN(K.layers.Layer):

    def __init__(self, units, edge_feature, dropout_rate=0.5, activation=tf.nn.relu):
        super(GCN, self).__init__()
        self.activation = activation
        self.edge_feature_num = edge_feature - 1
        self.atom_dense_list = [K.layers.Dense(units=units) for _ in range(self.edge_feature_num)]
        self.atom_dense = K.layers.Dense(units=units)
        self.dropout = K.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        adjacency_tensor, hidden_tensor, node_tensor = inputs
        adj = tf.transpose(adjacency_tensor[:, :, :, 1:], (0, 3, 1, 2))
        annotations = tf.concat((hidden_tensor, node_tensor), -1) if hidden_tensor is not None else node_tensor
        output = tf.stack([d(inputs=annotations) for d in self.atom_dense_list], 1)
        output = tf.matmul(adj, output)
        output = tf.reduce_sum(output, 1) + self.atom_dense(annotations)
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output, training=training)
        return output

    def get_config(self):
        base_config = super(GCN, self).get_config()
        base_config['output_dim'] = self.atom_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GraphAggregationLayer(K.layers.Layer):

    def __init__(self, units, activation=None, dropout_rate=0.5):
        super(GraphAggregationLayer, self).__init__()
        self.scale_factor_dense = K.layers.Dense(units=units, activation=tf.nn.sigmoid)
        self.value_dense = K.layers.Dense(units=units, activation=activation)
        self.activation = activation
        self.dropout = K.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        i = self.scale_factor_dense(inputs)
        j = self.value_dense(inputs)
        output = tf.reduce_sum(i * j, 1)
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output, training=training)
        return output


class MLP(K.layers.Layer):

    def __init__(self, units, activation=None, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.dense_list = [K.layers.Dense(u, activation=activation) for u in units]
        self.dropout_list = [K.layers.Dropout(dropout_rate) for _ in units]

    def call(self, inputs):
        hidden_tensor = inputs
        for i in range(len(self.dense_list)):
            hidden_tensor = self.dense_list[i](hidden_tensor)
            hidden_tensor = self.dropout_list[i](hidden_tensor)
        return hidden_tensor


class MGCN(K.layers.Layer):

    def __init__(self, units, edge_feature, dropout_rate=0.5, activation=tf.nn.relu):
        super(MGCN, self).__init__()
        self.gcn_list = [
            GCN(u, edge_feature=edge_feature, dropout_rate=dropout_rate, activation=activation) for u
            in units]

    def call(self, inputs, training=False):
        adjacency_tensor, hidden_tensor, node_tensor = inputs
        for gcn in self.gcn_list:
            hidden_tensor = gcn((adjacency_tensor, hidden_tensor, node_tensor), training=training)
        return hidden_tensor
