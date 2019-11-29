import networkx as nx
import glob
import pickle
import random
import numpy as np
from multiprocessing import Pool

def walker(args, sub_size_list, degree, G, node_list):
    ''''''
    subgraph_set = []
    for i in node_list:
        if degree[i] < 3:
            continue
        for k in range(sub_size_list[i]):
            # sub_node_num = random.randint(3, args.max_graph_size)
            sub_node_num = args.max_graph_size
            seta = 5 * sub_node_num
            tem_vis = [0 for j in range(args.node_num)]
            tem_node_set = set()
            sub_node_set = []
            tem_node_set.add(i)
            tem_vis[i] = 1
            # print(tem_node_set)
            while len(sub_node_set) < sub_node_num:
                # print(sub_node_num)
                choose_node = [i]
                if random.random()<0.1:
                    while tem_vis[choose_node[0]]:
                        choose_node = random.sample(node_list, 1)
                    tem_vis[choose_node[0]] = 1
                else:
                    choose_node = random.sample(tem_node_set, 1)
                    tem_node_set.remove(choose_node[0])
                sub_node_set.append(choose_node[0])
                if len(tem_node_set) < seta:
                    for j in G.neighbors(choose_node[0]):
                        if not tem_vis[j] and degree[j] >= 2:
                            tem_vis[j] = 1
                            tem_node_set.add(j)
                if (len(tem_node_set) <= 0):
                    break
            subgraph_set.append(sub_node_set)
    return subgraph_set

def read_one_graph(threadID, G_file_name, node_set, G_file_num):
    G = nx.Graph()
    cnt = 0
    G_file = open(G_file_name, 'r')
    for line in G_file:
        tem = line[:-1].split(' ')
        if len(tem) < 2:
            break
        x = int(tem[0])
        y = int(tem[1])
        if x not in node_set or y not in node_set:
            continue
        G.add_edge(x, y)
        cnt += 1
        print("\r读取第%d个图  %.4f" % (threadID, cnt/G_file_num), end=" ")
    return G

def read_graph(args, G_o_file_name, G_n_file_name):
    #读取t时间的图和t+1时间的图
    # node_set_o = []
    node_set_n = set()
    G_o_file = open(G_o_file_name, 'r')
    G_n_file = open(G_n_file_name, 'r')
    G_o_file_num = 0
    G_n_file_num = 0
    for line in G_o_file:
        tem = line[:-1].split(' ')
        if len(tem) < 2:
            break
        G_o_file_num += 1

    for line in G_n_file:
        tem = line[:-1].split(' ')
        if len(tem) < 2:
            break
        x = int(tem[0])
        y = int(tem[1])
        node_set_n.add(x)
        node_set_n.add(y)
        G_n_file_num += 1

    results = []
    pool = Pool(processes=2)
    results.append(
        pool.apply_async(read_one_graph, (1, G_o_file_name, node_set_n, G_o_file_num)))
    results.append(
        pool.apply_async(read_one_graph, (2, G_n_file_name, node_set_n, G_n_file_num)))
    pool.close()
    pool.join()
    print("网络读取完成")
    results = [res.get() for res in results]
    G_o = results[0]
    G_n = results[1]
    print("取出   图   线程数据")

    print("共有点 ： "+str(len(node_set_n)))
    return [G_o, G_n]

def read_dynamic_graph(args, G_files):
    G_list = []
    for i in range(len(G_files)-1):
        if len(G_files) == 2:
            G_list.append(read_graph(args, G_files[i], G_files[i+1]))
        else:
            G_list.append(read_graph(args, G_files[i], G_files[i]))
    return G_list

def get_sub_size(args, G):
    if args.type != 'dynamic':
        node_num = len(G.nodes())
        size_list = [0 for i in range(args.node_num)]
        degree = [0 for i in range(args.node_num)]
        for edge in G.edges():
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        # print(degree)
        ds = [0 for i in range(args.node_num)]
        k = np.log10(node_num)
        for v in G.nodes():
            ds[v] += degree[v]
            for node in G.neighbors(v):
                ds[v] += degree[node]
            size_list[v] = int(args.max_each*(degree[v]/ds[v]+1/(1+np.exp(-degree[v]/k))))
        return size_list, degree
    else:
        dynamic_size_list = []
        dynamic_degree = []
        for G_s in G:
            G_s = G_s[0]
            node_num = len(G_s.nodes())
            size_list = [0 for i in range(args.node_num)]
            degree = [0 for i in range(args.node_num)]
            for edge in G_s.edges():
                # print(edge[0])
                # print(edge[1])
                degree[edge[0]] += 1
                degree[edge[1]] += 1
            # print(degree)
            ds = [0 for i in range(args.node_num)]
            k = np.log10(node_num)
            for v in G_s.nodes():
                ds[v] += degree[v]
                for node in G_s.neighbors(v):
                    ds[v] += degree[node]
                size_list[v] = int(args.max_each * (degree[v] / ds[v] + 1 / (1 + np.exp(-degree[v] / k))))
            dynamic_size_list.append(size_list)
            dynamic_degree.append(degree)
        return dynamic_size_list, dynamic_degree

def generate_train_data(args, type):
    if type == 'train':
        G_o_file_name = 'data/' + args.dataset + '/0000.inp'
        G_n_file_name = 'data/' + args.dataset + '/0000.inp'
        G_o, G_n = read_graph(args, G_o_file_name, G_n_file_name)
    else:
        G_o_file_name = 'data/' + args.dataset + '/0000.inp'
        G_n_file_name = 'data/' + args.dataset + '/0001.inp'
        G_o, G_n = read_graph(args, G_o_file_name, G_n_file_name)

    subgraph_set = []
    print("准备游走，生成点集上限")
    sub_size_list, degree = get_sub_size(args, G_o)
    # print(sub_size_list)
    print("开始子图生成游走")
    node_list = list(G_o.nodes())
    per_threads_node = len(node_list)//args.walkers
    #创建新线程
    results = []
    pool = Pool(processes=args.walkers)
    for i in range(args.walkers):
        if i == args.walkers - 1:
            results.append(pool.apply_async(walker, (args, sub_size_list, degree, G_o, node_list[per_threads_node*i:])))
        else:
            results.append(pool.apply_async(walker, (args, sub_size_list, degree, G_o, node_list[per_threads_node*i:per_threads_node*(i+1)])))
    pool.close()
    pool.join()
    print("所有游走完成")
    results = [res.get() for res in results]
    # print(len(results))
    for it in results:
        for jk in it:
            subgraph_set.append(jk)
    print("取出   游走   线程数据")
    return subgraph_set, [G_o, G_n]


def generate_dynamic_train_data(args, type):
    if type == 'train':
        G_files = 'dynamic-data/' + args.dataset + '/*.inp'
        files = glob.glob(G_files)
        G_list = read_dynamic_graph(args, files)
    else:
        G_files = 'dynamic-data/' + args.dataset + '/*.inp'
        files = glob.glob(G_files)
        # print(files[-2:])
        G_list = read_dynamic_graph(args, files[-2:])

    subgraph_set = []
    print("准备游走，生成点集上限")
    sub_size_list, degree = get_sub_size(args, G_list)
    # print(sub_size_list)
    print("开始 动态 子图生成游走")
    cn = 0
    for k, G_par in enumerate(G_list):
        G_o, G_n = G_par
        sub_subgraph_set = []
        cn += 1
        node_list = list(G_o.nodes())
        per_threads_node = len(node_list)//args.walkers
        #创建新线程
        results = []
        pool = Pool(processes=args.walkers)
        for i in range(args.walkers):
            if i == args.walkers - 1:
                results.append(pool.apply_async(walker, (args, sub_size_list[k], degree[k], G_o, node_list[per_threads_node*i:])))
            else:
                results.append(pool.apply_async(walker, (args, sub_size_list[k], degree[k], G_o, node_list[per_threads_node*i:per_threads_node*(i+1)])))
        pool.close()
        pool.join()
        print("第 %d 个图游走完成"%cn)
        # print(results)
        results = [res.get() for res in results]
        # print(len(results))
        for it in results:
            for jk in it:
                sub_subgraph_set.append(jk)
        print("取出   游走   线程数据")
        subgraph_set.append(sub_subgraph_set)
    return subgraph_set, G_list