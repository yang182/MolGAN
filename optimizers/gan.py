import tensorflow as tf
import numpy as np
from utils.utils import reward


class GraphGANOptimizer(object):

    def __init__(self, model, learning_rate=1e-3, feature_matching=True):
        self.la = tf.placeholder_with_default(1., shape=())

        with tf.name_scope('losses'):
            eps = tf.random_uniform(tf.shape(model.logits_real)[:1], dtype=model.logits_real.dtype)

            x_int0 = model.adjacency_tensor * tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1),
                                                             -1) + model.edges_softmax * (
                             1 - tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1))
            x_int1 = model.node_tensor * tf.expand_dims(tf.expand_dims(eps, -1), -1) + model.nodes_softmax * (
                    1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))

            grad0, grad1 = tf.gradients(model.D_x((x_int0, None, x_int1), model.discriminator_units), (x_int0, x_int1))

            self.grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1)) + tf.reduce_mean(
                ((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keep_dims=True)

            self.loss_D = - model.logits_real + model.logits_fake
            self.loss_G = - model.logits_fake
            self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + (
                    model.value_logits_fake - model.rewardF) ** 2
            self.loss_RL = - model.value_logits_fake
            self.loss_F = (tf.reduce_mean(model.features_real, 0) - tf.reduce_mean(model.features_fake, 0)) ** 2

        self.loss_D = tf.reduce_mean(self.loss_D)
        self.loss_G = tf.reduce_sum(self.loss_F) if feature_matching else tf.reduce_mean(self.loss_G)
        self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        alpha = tf.abs(tf.stop_gradient(self.loss_G / self.loss_RL))
        self.grad_penalty = tf.reduce_mean(self.grad_penalty)

        with tf.name_scope('train_step'):
            self.train_step_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_D + 10 * self.grad_penalty,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

            self.train_step_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_G, lambda: 0.) + tf.cond(
                    tf.less(self.la, 1), lambda: (1 - self.la) * alpha * self.loss_RL, lambda: 0.),
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

            self.train_step_V = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_V,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))


def train_step(data, model, la=1.0, train_g=True, feature_matching=True, batch_dim=8):
    mols, _, _, data_a, data_x, _, _, _, _ = data.next_validation_batch(batch_size=batch_dim)
    with tf.GradientTape(persistent=True) as grad_tape:
        v = model((data_a, data_x), training=True)
        real_sample, gen_sample, gen_raw = v
        adjacency_tensor, node_tensor, logits_real, features_real, value_logits_real = real_sample
        edges_hat, nodes_hat, logits_fake, features_fake, value_logits_fake = gen_sample
        (edges_softmax, nodes_softmax, edges_argmax, nodes_argmax, edges_gumbel_logits, nodes_gumbel_logits,
         edges_gumbel_softmax, nodes_gumbel_softmax, edges_gumbel_argmax, nodes_gumbel_argmax) = gen_raw

        n, e = nodes_gumbel_argmax, edges_gumbel_argmax
        n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
        gen_mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
        rewardF = reward(gen_mols)  # 生成分子
        rewardR = reward(mols)  # 采样分

        la = tf.constant(la)
        # model.logits_real (batch, 1)

        eps = tf.random.uniform(tf.shape(logits_real)[:1], dtype=logits_real.dtype)
        # model.adjacency_tensor (batch, 9, 9, 5)
        # model.edges_softmax (8, 9, 9, 5)
        x_int0 = adjacency_tensor * tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1),
                                                   -1) + edges_softmax * (
                         1 - tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1))

        x_int1 = node_tensor * tf.expand_dims(tf.expand_dims(eps, -1), -1) + nodes_softmax * (
                1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))
        with tf.GradientTape(persistent=True) as g:
            g.watch(x_int0)
            g.watch(x_int1)
            grad0, grad1 = g.gradient(model.D_x((x_int0, None, x_int1)), (x_int0, x_int1))
        grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1)) + tf.reduce_mean(
            ((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keepdims=True)
        # print("$" * 200)
        # print("gradient penalty: ")
        # print(grad_penalty)
        # print("$" * 200)

        loss_D = - logits_real + logits_fake
        loss_G = - logits_fake
        loss_V = (value_logits_real - rewardR) ** 2 + (value_logits_fake - rewardF) ** 2
        loss_RL = - value_logits_fake
        loss_F = (tf.reduce_mean(features_real, 0) - tf.reduce_mean(features_fake, 0)) ** 2

        loss_D = tf.reduce_mean(loss_D)
        loss_G = tf.reduce_sum(loss_F) if feature_matching else tf.reduce_mean(loss_G)
        loss_V = tf.reduce_mean(loss_V)
        loss_RL = tf.reduce_mean(loss_RL)
        alpha = tf.abs(tf.stop_gradient(loss_G / loss_RL))
        grad_penalty = tf.reduce_mean(grad_penalty)

        loss_d = loss_D + 10 * grad_penalty
        loss_g = tf.cond(tf.greater(la, 0), lambda: la * loss_G, lambda: 0.) + tf.cond(
                    tf.less(la, 1), lambda: (1 - la) * alpha * loss_RL, lambda: 0.)
        loss_v = loss_V

    optimizer = tf.optimizers.Adam(learning_rate=model.learning_rate)

    if train_g:
        grad_g = grad_tape.gradient(loss_g, model.G_x.trainable_variables)    # generator
        optimizer.apply_gradients(zip(grad_g, model.G_x.trainable_variables))
        # print("gradient g: ")
        # print("*=" * 100)
        # print(grad_g)
        # print("*=" * 100)
    else:
        grad_d = grad_tape.gradient(loss_d, model.D_x.trainable_variables)     # discriminator
        optimizer.apply_gradients(zip(grad_d, model.D_x.trainable_variables))
        # print("*"*200)
        # print("gradient d: ")
        # print(grad_d)
        # print("*"*200)
    if train_g and la < 1:
        grad_v = grad_tape.gradient(loss_v, model.V_x.trainable_variables)      # value
        optimizer.apply_gradients(zip(grad_v, model.V_x.trainable_variables))
        # print("=" * 200)
        # print("gradient g: ")
        # print(grad_v)
        # print("=" * 200)

    return loss_d, loss_g, loss_v
