import pickle
import numpy as np
import glob
import generator
import os
from sklearn import preprocessing
import random
import tensorflow as tf
from modules import label_smoothing
from scipy import sparse
from multiprocessing import Pool

def spliter(subgraph_set, Go, Gn):
    inpu = []
    pro = []
    output = []
    # nodeset_o = list(Go.nodes())
    # nodeset_n = list(Gn.nodes())
    for sub_set in subgraph_set:
        tem_pro = []
        tem_out = []
        sub_size = len(sub_set)
        sub_size *= sub_size
        for i, node_1 in enumerate(sub_set):
            pro_vec = []
            for j, node_2 in enumerate(sub_set):
                # if node_1 not in nodeset_o:
                #     continue
                if node_2 in Go.neighbors(node_1):
                    # print(j," ",rem2origin[node_1]," ",rem2origin[node_2])
                    pro_vec.append(node_2)
                else:
                    pro_vec.append(0)
                    # pro_vec[j] = 1.0
            # print(len(sub_set))
            # pro_vec.extend([0 for i in range(max_graph_size - len(pro_vec))])
            tem_pro.append(pro_vec)
        cou = 0
        for i, node_1 in enumerate(sub_set):
            # 为了输入输出需要，固定成和embdeding一样的维度
            # pro_out = np.zeros((len(sub_set)))
            pro_out = [0 for i in range(len(sub_set))]
            pro_out = np.array(pro_out)
            pro_out[i] = 1.0
            for j, node_2 in enumerate(sub_set):
                # if node_1 not in nodeset_n:
                #     continue
                if node_2 in Gn.neighbors(node_1):
                    cou += 1
                    pro_out[j] = 1.0
            # 对标签输出进行平滑处理
            # pro_out = label_smoothing(pro_out)
            # 对标签输出进行正则化处理
            # pro_vec = preprocessing.normalize(pro_vec, norm='l2')
            tem_out.append(pro_out)
        if cou/sub_size < 0.3:
        #     # print(cou/sub_size)
            continue
        inpu.append(sub_set)
        pro.append(tem_pro)
        output.append(tem_out)
        # print(pro[0])
    return [inpu, pro, output]

def convert_2_str(x, dimension):
    str_x = []
    if dimension == 3:
        for k, x_i in enumerate(x):
            tem = ''
            for j, it in enumerate(x_i):
                for i, t in enumerate(it):
                    if i == 0:
                        tem = str(t)
                    else:
                        tem = tem + '*' + str(t)
                tem = tem + '_'
            tem = tem + '3'
            str_x.append(tem)
    else:
        for x_i in x:
            tem = ''
            row = len(x_i)
            for it in x_i:
                if dimension == 2:
                    for y in it:
                        tem = tem + str(y) + '_'
                else:
                    tem = tem + str(it) + '_'

            tem = tem + str(row) + '_'
            if dimension == 2:
                tem = tem + str(len(x_i[0])) + '_'
            tem = tem + str(dimension)
            str_x.append(tem)
    return str_x

def convert_2_arr(x):
    arr_x = []
    # print(x)
    tem = x.split("_")
    dimension = tem[-1]
    if dimension == '1':
        row = int(tem[-2])
    elif dimension == '2':
        row = int(tem[-3])
        col = int(tem[-2])
    if dimension == '1':
        for it in tem[:row]:
            if it == '':
                continue
            arr_x.append(int(it))
    elif dimension == '2':
        pos = 0
        for i in range(row):
            tem_col = []
            for j in range(col):
                # print(pos)
                tem_col.append(float(tem[pos]))
                pos += 1
            arr_x.append(tem_col)
        return np.array(arr_x), row
    else:
        MAX = 0
        for arr in tem[:-1]:
            each_line = arr.split('*')
            one_row = []
            # print(each_line)
            for it in each_line:
                if it == '':
                    continue
                one_row.append(int(it))
            MAX = max(MAX, len(one_row))
            arr_x.append(one_row)
        for i in range(len(arr_x)):
            for j in range(MAX - len(arr_x[i])):
                arr_x[i].append(0)
        return np.array(arr_x), 0
    # print(x)
    # print(arr_x)
    return np.array(arr_x), row

def generator_fn(input, pro, output):
    for input_, output_, pro_ in zip(input, output, pro):
        input_, x_seqlen = convert_2_arr(input_.decode())
        # print(input_.shape)

        output_, y_seqlen = convert_2_arr(output_.decode())
        # print(output_.shape)
        pro_, _ = convert_2_arr(pro_.decode())
        # print(pro_.shape)
        yield (input_, x_seqlen), (pro_, output_, y_seqlen)

def train_data(args, type):
    if args.type == 'dynamic':
        graph_filename = 'dynamic-train/'+args.dataset+'_'+type+'.dat'
        if not os.path.exists(graph_filename):
            subgraph_set, G_list = generator.generate_dynamic_train_data(args, type)
            pickle.dump([subgraph_set, G_list], open(graph_filename, 'wb'))
        else:
            print("已经存在数据，开始读取-----动态数据")
            subgraph_set, G_list = pickle.load(open(graph_filename, 'rb'))
        if type == 'train':
            print("开始生成训练数据")
        else:
            print("开始生成测试数据")
        # print(subgraph_set)
        sub = []
        pro = []
        output = []
        for s, sub_set in enumerate(subgraph_set):
            subset_size = len(sub_set)
            per_spliter_each = subset_size // args.spliter
            results = []
            pool = Pool(processes=args.spliter)
            for i in range(args.spliter):
                if i == args.spliter - 1:
                    results.append(pool.apply_async(spliter, (sub_set[per_spliter_each * i:], G_list[s][0], G_list[s][1])))
                else:
                    results.append(
                        pool.apply_async(spliter,
                                         (sub_set[per_spliter_each * i:per_spliter_each * (i + 1)], G_list[s][0], G_list[s][1])))
            pool.close()
            pool.join()

            results = [res.get() for res in results]
            s_sub = []
            s_pro = []
            s_output = []
            for i in range(args.spliter):
                s_sub += results[i][0]
                s_pro += results[i][1]
                s_output += results[i][2]
            sub.extend(s_sub)
            pro.extend(s_pro)
            output.extend(s_output)
            print("第 %d 批训练数据完成" % s)
        print("生成训练数据完成")

    else:
        graph_filename = 'train/' + args.dataset + '_' + type + '.dat'
        if not os.path.exists(graph_filename):
            subgraph_set, G_list = generator.generate_train_data(args, type)
            pickle.dump([subgraph_set, G_list], open(graph_filename, 'wb'))
        else:
            print("已经存在数据，开始读取-----静态数据")
            subgraph_set, G_list = pickle.load(open(graph_filename, 'rb'))
        subset_size = len(subgraph_set)
        per_spliter_each = subset_size // args.spliter
        results = []
        pool = Pool(processes=args.spliter)
        if type == 'train':
            print("开始生成训练数据")
        else:
            print("开始生成测试数据")
        for i in range(args.spliter):
            if i == args.spliter - 1:
                results.append(pool.apply_async(spliter, (subgraph_set[per_spliter_each * i:], G_list[0], G_list[1])))
            else:
                results.append(
                    pool.apply_async(spliter,
                                     (subgraph_set[per_spliter_each * i:per_spliter_each * (i + 1)], G_list[0], G_list[1])))
        pool.close()
        pool.join()

        print("生成训练数据完成")
        results = [res.get() for res in results]
        sub = []
        pro = []
        output = []
        for i in range(args.spliter):
            sub += results[i][0]
            pro += results[i][1]
            output += results[i][2]
        # print(pro[0])

    # emb = graph_emb(args, emb_filename)
    # tran_ma = get_global_transfer_matrix(args, G_o)

    #生成训练集

    print("取出   训练    线程数据")

    #None是变长，其他维数固定
    shapes = (([None], ()),
              ([None, None], [None, None], ()))
    padded_shapes = (([None], ()),
              ([None, None], [None, None], ()))
    types = ((tf.int32, tf.int32),
             (tf.int32, tf.float32, tf.int32))
    # print(pro)
    #多线程处理数据到字符串

    input_set = convert_2_str(sub, 1)
    pro_matrix = convert_2_str(pro, 2)
    output_set = convert_2_str(output, 2)

    number_samples = len(input_set)
    print("共有训练数据集:  ", number_samples)
    #这里也要改
    # emblen = tf.constant([1, args.node_num])
    input_dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(input_set, pro_matrix, output_set))  # <- arguments 必须是定长的，不可以是不定长的list

    input_dataset = input_dataset.repeat()  # iterate forever
    input_dataset = input_dataset.padded_batch(args.batch_size, padded_shapes, padding_values=None, drop_remainder=True).prefetch(1)

    # eval_dataset = eval_dataset.repeat(10)  # iterate forever
    # eval_dataset = eval_dataset.padded_batch(args.batch_size, padded_shapes, padding_values=None, drop_remainder=True).prefetch(1)

    results = [number_samples // args.batch_size + (number_samples % args.batch_size != 0), number_samples]
    # return input_dataset, emb, results
    return input_dataset, results

def graph_emb(args, emb_filename):
    try:
        emb = [[] for i in range(args.node_num)]
        f = open(emb_filename, 'r')
        f.readline()
        for line in f.readlines():
            tem = line.split(' ')
            if len(tem)<2:
                break
            node = int(tem[0])
            for it in tem[1:]:
                emb[node].append(float(it))
        emb = tf.convert_to_tensor(emb)
        return emb
    except IOError:
        print("fail to initialize the embedding.")

def get_global_transfer_matrix(args, G):
    edges_set = set(G.edges())
    # edges = list(G.edges())
    indr = []
    indc = []
    data = np.zeros((len(edges_set)))
    for i, edge in enumerate(edges_set):
        indr.append(edge[0])
        indc.append(edge[1])
        # print(G.get_edge_data(edge[0], edge[1]))
        data[i] = len(G.get_edge_data(edge[0], edge[1]))
    #用行切片矩阵进行处理

    adj_ma = sparse.csr_matrix((data, (indr, indc)), shape=(args.node_num, args.node_num))
    # print(adj_ma.toarray().shape)
    hop_ma = adj_ma.copy()
    tran_ma = adj_ma.copy()
    hy = 0.5
    for i in range(args.k_hop):
        hop_ma = hop_ma*adj_ma
        tran_ma = tran_ma + hy*hop_ma
        hy /= 2
    #只能用这个函数，array里面的A*B是对应位置相乘=np.multiply()，不是矩阵乘法np.dot()
    one_vec = np.asarray(tran_ma.sum(axis=1).T)
    # print(one_vec.shape)
    indr = [i for i in range(args.node_num)]
    indc = [i for i in range(args.node_num)]
    data = [1.0/(one_vec[0][i]+1.0) for i in range(args.node_num)]
    # print(data)
    dia_ma = sparse.csr_matrix((data, (indr, indc)), shape=(args.node_num, args.node_num))
    # print(dia_ma)
    tran_ma = np.dot(dia_ma, tran_ma)

    return tran_ma