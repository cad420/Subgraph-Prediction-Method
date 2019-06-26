# -*- coding: utf-8 -*-
#/usr/bin/python3
import sys
sys.path.append("/")
import tensorflow as tf
from model import Transformer
from tqdm import tqdm
from load_data import train_data
from utils import save_hparams, save_variable_specs
import os
from hparams import Hparams
import math
import logging

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main():
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    save_hparams(hp, hp.logdir)
    print("开始读取数据")
    input_set,  results = train_data(hp, 'train')
    eval_input_set,  results_eval = train_data(hp, 'evaluation')
    num_train_batches, num_train_samples = results
    num_eval_batches, num_eval_samples = results_eval
    print("读取数据完成")

    iter = tf.data.Iterator.from_structure(input_set.output_types, input_set.output_shapes)
    xs, ys = iter.get_next()

    train_init_op = iter.make_initializer(input_set)
    eval_init_op = iter.make_initializer(eval_input_set)
    print("构建模型")
    m = Transformer(hp)
    loss, train_op, global_step, train_summaries = m.train(xs, ys)
    if hp.type == 'attribute':
        accuracy = m.eval(xs, ys)
    else:
        exist_pre, no_exist_pre, all_pre = m.eval(xs, ys)
    print("开始训练")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_op)
        total_steps = hp.num_epochs * num_train_batches
        _gs = sess.run(global_step)
        for i in tqdm(range(_gs, total_steps+1)):
            _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
            epoch = math.ceil(_gs / num_train_batches)
            if _gs and _gs % num_train_batches == 0:
                _loss = sess.run(loss) # train loss
                _ = sess.run([eval_init_op])
                if hp.type == 'attribute':
                    acc = sess.run([accuracy])
                    print("预测准确率为：  ", acc)
                else:
                    pre, no_pre, al_pre = sess.run([exist_pre, no_exist_pre, all_pre])
                    print("\n有边的预测准确率为：  ", pre)
                    print("无边的预测准确率为：  ", no_pre)
                    print("综合预测准确率为：  ", al_pre)
                print("Epoch : %02d   loss : %.2f" % (epoch, _loss))
                sess.run(train_init_op)

if __name__ == '__main__':
    main()