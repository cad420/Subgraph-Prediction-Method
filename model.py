# -*- coding: utf-8 -*-
# /usr/bin/python3
'''

'''
import tensorflow as tf
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme, bivalue, get_diag
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
    # print(y[0])
    return x

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        输入的每个序列的实际长度
        x_seqlens: int32 tensor. (N,)
        输入的实际的句子
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        输出的实际长度，两个都一样的
        y_seqlen: int32 tensor. (N, )
        输出的实际的句子
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        # indicator dir
        # self.token2idx, self.idx2token = load_vocab(hp.vocab)
        # pre-trained embedding
        self.embeddings = get_token_embeddings(self.hp.node_num, self.hp.d_model, zero_pad=True)
        # self.tran_ma = emb

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        # 实现参数共享，在多个encoder里传递参数
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens = xs
            # print(x)
            # embedding
            #通过对应的索引直接变成embedding，这里用的就是点的embedding
            # print(x.shape.as_list())
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
    ############################################################################################
            #改用全局邻接矩阵来获取信息
            # print(enc.shape.as_list())
            # enc = tf.nn.embedding_lookup(self.tran_ma, x)
            # print(enc.shape.as_list())
            # enc = tf.transpose(enc, [0, 2, 1])
            # print(enc.shape.as_list())
            # enc = tf.nn.embedding_lookup(enc, x)
            # print(enc.shape.as_list())
            # print(enc)
            # enc = tf.py_func(func, [self.tran_ma, x, seqlens, self.hp.d_model], tf.float32)
            # print(enc.shape.as_list())
            # enc *= self.hp.d_model**0.5 # scale
            # print(self.hp.maxlen1)
    #############################################################################################
            enc += positional_encoding(enc, self.hp.maxlen1, self.hp)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory

    def decode(self, xs, ys, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            x, seqlens = xs
            decoder_inputs, y, seqlens = ys
            # embedding
            # dec = tf.matmul(decoder_inputs, tf.tile(tf.expand_dims(self.embeddings, 0), multiples=[128, 1, 1]))
            # dec = tf.matmul(decoder_inputs, self.embeddings)  # (N, T2, d_model)

            dec = tf.reduce_sum(tf.nn.embedding_lookup(self.embeddings, decoder_inputs), reduction_indices=2) # (N, T1, d_model)
            # test_dec = dec
            dec = dec * self.hp.d_model ** 0.5  # scale
            #子图结构里也需要对应的位置编码，因为要对应输出的预测结构
            dec += positional_encoding(dec, self.hp.maxlen2, self.hp)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              #是否加上mask层
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        # dec = tf.layers.dense(inputs=dec, units=self.hp.d_model, activation=tf.nn.relu, trainable=True)
        # logits = tf.layers.dense(inputs=dec, units=self.hp.d_model, activation=tf.nn.relu, trainable=True)
        weights = tf.nn.embedding_lookup(self.embeddings, x) # (d_model, T2)
        logits = tf.einsum('ntd,nkd->ntk', dec, weights) # (N, T2, T2)
        logits = (logits + tf.transpose(logits, [0, 2, 1]))/2 #强制最终结果为一个对称矩阵，符合
##########################################################################################需要修改输出的维度
        # weights = tf.nn.embedding_lookup(self.embeddings, x)
        # diag_a = 1/tf.sqrt(tf.reduce_sum(dec*dec, axis=-1))
        # diag_b = 1/tf.sqrt(tf.reduce_sum(weights*weights, axis=-1))
        # logits = tf.einsum('nij,njk,ni,nk->nik', dec, tf.transpose(weights, [0, 2, 1]), diag_a, diag_b)  # (N, T2, T2)
        #用sigmod找出存在的边
        # logits = tf.to_int32(tf.sigmoid(logits))
        return logits, y, decoder_inputs, weights

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory = self.encode(xs)
        logits, y_, dec, weight = self.decode(xs, ys, memory)

        # train scheme
        #需要进行label smoothing么？？？？？？？？？？？
        # y_ = label_smoothing(y)
        #更换后的cross_entropy，适用于多label的情况
        # ce = tf.nn.sigmoid_cross_entropy_with_logits (logits=logits, targets=y_)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_)
        loss = tf.reduce_sum(ce)
        # loss = tf.reduce_sum(tf.square(logits-y_))
        #loss函数的位置
        # loss = tf.reduce_mean(tf.abs(logits - y_))
        # print(ce)
        # a = tf.py_func(test, [logits, y_], tf.float32)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):#修改这个验证的函数
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        #需要用到结构矩阵
        decoder_inputs, y, y_seqlen = ys

        # decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        # ys = (decoder_inputs, y, y_seqlen)

        memory = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        # for _ in tqdm(range(self.hp.maxlen2)):
        #     logits, y_hat, y = self.decode(ys, memory, False)
        #     #需要去掉
        #     # if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break
        #
        #     _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
        #     ys = (_decoder_inputs, y, y_seqlen)

        # monitor a random sample
        # n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        # sent1 = sents1[n]
        # pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)

        logits, y, dec, weight = self.decode(xs, ys, memory)
        # print(logits)
        # print(y)
        exist_pre, no_exist_pre, all_pre= tf.py_func(bivalue, [logits, y, dec, weight], [tf.double, tf.double, tf.double], stateful=True)
        summaries = tf.summary.merge_all()
        #需要加上计算loss的过程，或者各种其他的东西
        return exist_pre, no_exist_pre, all_pre, summaries

