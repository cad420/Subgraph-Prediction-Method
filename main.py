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
    # logging.basicConfig(level=logging.INFO)
    # logging.info("# hparams")
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    save_hparams(hp, hp.logdir)
    # logging.info("# Prepare train/eval batches")
    print("开始读取数据")
    input_set,  results = train_data(hp, 'train')
    eval_input_set,  results_eval = train_data(hp, 'evaluation')
    num_train_batches, num_train_samples = results
    num_eval_batches, num_eval_samples = results_eval
    print("读取数据完成")
    # input_set, pro_matrix, output_set = get_batch(hp, input_set, pro_matrix, output_set) #将训练数据集分批次
    # print(input_set.output_types)
    # print(input_set.output_shapes)


    iter = tf.data.Iterator.from_structure(input_set.output_types, input_set.output_shapes)
    xs, ys = iter.get_next()

    # print(input_set.output_types)
    # print(input_set.output_shapes)
    train_init_op = iter.make_initializer(input_set)
    eval_init_op = iter.make_initializer(eval_input_set)
    print("构建模型")
    # logging.info("# Load model")
    m = Transformer(hp)
    loss, train_op, global_step, train_summaries = m.train(xs, ys)
    exist_pre, no_exist_pre, all_pre,eval_summaries = m.eval(xs, ys)
    # y_hat = m.infer(xs, ys)

    # logging.info("# Session")
    # saver = tf.train.Saver(max_to_keep=hp.num_epochs)
    print("开始训练")
    with tf.Session() as sess:
        # ckpt = tf.train.latest_checkpoint(hp.logdir)
        # print("okokokokokokokok")
        # if ckpt is None:
        #     logging.info("Initializing from scratch")
        #     sess.run(tf.global_variables_initializer())
        #     save_variable_specs(os.path.join(hp.logdir, "specs"))
        # else:
        #     saver.restore(sess, ckpt)
        # logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        # save_variable_specs(os.path.join(hp.logdir, "specs"))

        # summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

        sess.run(train_init_op)
        total_steps = hp.num_epochs * num_train_batches
        _gs = sess.run(global_step)
        for i in tqdm(range(_gs, total_steps+1)):
            # print("okokokokokokokok")
            _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
            # print("qwqwqwqwqwqwqwqw")
            epoch = math.ceil(_gs / num_train_batches)
            # summary_writer.add_summary(_summary, _gs)

            if _gs and _gs % num_train_batches == 0:
                # logging.info("epoch {} is done".format(epoch))
                _loss = sess.run(loss) # train loss

                # logging.info("# test evaluation")
                _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
                pre, no_pre, al_pre = sess.run([exist_pre, no_exist_pre, all_pre])
                print("\n有边的预测准确率为：  ", pre)
                print("无边的预测准确率为：  ", no_pre)
                print("综合预测准确率为：  ", al_pre)
                # summary_writer.add_summary(_eval_summaries, _gs)

                # logging.info("# write results")
                model_output = hp.dataset+"_E%02dL%.2f" % (epoch, _loss)
                print("Epoch : %02d   loss : %.2f" % (epoch, _loss))

                # logging.info("# save models")
                # ckpt_name = os.path.join(hp.logdir, model_output)
                # saver.save(sess, ckpt_name, global_step=_gs)
                # logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

                # logging.info("# fall back to train mode")
                sess.run(train_init_op)
        # summary_writer.close()
    # logging.info("Done")

if __name__ == '__main__':
    main()