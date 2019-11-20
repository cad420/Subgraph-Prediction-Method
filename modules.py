# -*- coding: utf-8 -*-
# /usr/bin/python3

import numpy as np
import tensorflow as tf


def ln(inputs, epsilon=1e-8, scope="ln"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def get_init_embeddings(vocab_size, num_units, zero_pad=True):
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
        outputs /= d_k ** 0.5

        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs



def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)


        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        outputs += queries
        outputs = ln(outputs)

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs
        outputs = ln(outputs)

    return outputs

def positional_encoding(inputs,
                        maxlen,
                        hp,
                        masking=False,
                        scope="positional_encoding"):
    # print(type(maxlen))
    E = hp.d_model  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        return tf.to_float(outputs)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def bivalue(logist, label, dec):
    logist = 1 / (1 + np.exp(-logist))
    logist[logist > 0.5] = 1.0
    logist[logist <= 0.5] = 0

    exist_total = 0
    no_exist_total = 0
    exist_label = 0
    no_exist_label = 0
    for n in range(logist.shape[0]):
        for i in range(logist[n].shape[0]):
            for j in range(logist[n][i].shape[0]):
                # total += 1
                # if abs(logist[n][i][j] - label[n][i][j]) < 0.00001:
                #     true_label += 1
                if label[n][i][j] == 1:
                    exist_total += 1
                    if logist[n][i][j] == label[n][i][j]:
                        exist_label += 1
                else:
                    no_exist_total += 1
                    if logist[n][i][j] == label[n][i][j]:
                        no_exist_label += 1
    # print(true_label)
    # print(total)
    return exist_label/exist_total, no_exist_label/no_exist_total, (exist_label+no_exist_label)/(exist_total+no_exist_total)

def biclass(logist, label):
    logist[logist > 0.5] = 1.0
    logist[logist <= 0.5] = 0
    ans = 0
    cnt = 0
    for n in range(logist.shape[0]):
        for i in range(logist[n].shape[0]):
            if label[cnt] == logist[n][i]:
                ans += 1
            cnt += 1
    return ans / cnt

def get_diag(data, embedding):
    result = []
    # print(data)
    # print(embedding)
    for mat in data:
        diag = np.diag(mat)
        diag_mat = []
        for i in range(mat.shape[0]):
            diag_mat.append(diag*diag[i])
        diag_mat = np.array(diag_mat)
        diag_mat = 1/diag_mat
        result.append(diag_mat)
    # print(result)
    return np.array(result)