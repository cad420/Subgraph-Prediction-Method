# -*- coding: utf-8 -*-
# /usr/bin/python3
import tensorflow as tf
from modules import get_init_embeddings, ff, positional_encoding, multihead_attention, noam_scheme, bivalue, biclass
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

def func(tran, x, seqlen, d_model):
    # return x
    enc = []
    for batch in x:
        sub_size = len(batch)
        one_hot = np.zeros((sub_size, tran.shape[0]))
        for i, it in enumerate(x):
            one_hot[i][it] = 1.0
        tem = one_hot.dot(tran).dot(one_hot.T)
        tem = np.pad(tem, [[0, 0], [0, d_model-seqlen]])
        enc.append(tem)
    return np.array(enc)

def test(x, y):
    # print(1 / (1 + np.exp(-x[0])))
    # print(y[0]
    return x

class SENN:
    def __init__(self, hp):
        self.hp = hp
        self.embeddings = get_init_embeddings(self.hp.node_num, self.hp.d_model, zero_pad=True)

    def encode_decode(self, xs, ys, training=True):
        x, seqlens = xs
        decoder_inputs, y, seqlens = ys
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc = tf.nn.embedding_lookup(self.embeddings, x)  # (N, T1, d_model)
            enc += positional_encoding(enc, self.hp.maxlen1, self.hp)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            dec = tf.reduce_sum(tf.nn.embedding_lookup(self.embeddings, decoder_inputs),
                                reduction_indices=2)  # (N, T1, d_model)
            # test_dec = dec
            dec = dec * self.hp.d_model ** 0.5  # scale
            # 子图结构里也需要对应的位置编码，因为要对应输出的预测结构
            dec += positional_encoding(dec, self.hp.maxlen2, self.hp)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

        ## Blocks
        for i in range(self.hp.num_blocks):
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=dec,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              # 是否加上mask层
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        if self.hp.type == 'attribute':
            enc = tf.reduce_sum(enc, reduction_indices=1)
            dec = tf.reduce_sum(dec, reduction_indices=1)
            logits = tf.layers.dense(inputs=tf.concat(enc, dec), units=1, activation=tf.nn.relu)
        else:
            logits = tf.einsum('ntd,nkd->ntk', dec, enc) # (N, T2, T2)
            logits = (logits + tf.transpose(logits, [0, 2, 1]))/2 #强制最终结果为一个对称矩阵，符合
        return logits, y, decoder_inputs

    def train(self, xs, ys):
        logits, y_, dec = self.encode_decode(xs, ys)

        ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_)
        loss = tf.reduce_sum(ce)


        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        logits, y, dec = self.encode_decode(xs, ys)
        if self.hp.type == 'attribute':
            accuracy = tf.py_func(biclass, [logits, y, dec], [tf.double], stateful=True)
            return accuracy
        else:
            exist_pre, no_exist_pre, all_pre = tf.py_func(bivalue, [logits, y, dec], [tf.double, tf.double, tf.double], stateful=True)
            return exist_pre, no_exist_pre, all_pre

