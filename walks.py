from multiprocessing import Process
import random

class walker(Process):
    def __init__(self, threadID, args, sub_size_list, degree, G, node_list):
        Process.__init__(self)
        self.threadID = threadID
        self.subgraph_set = []
        self.hp = args
        self.sub_size_list = sub_size_list
        self.degree = degree
        self.G = G
        self.node_list = node_list
    def run(self):
        for i in self.node_list:
            if self.degree[i] < 3:
                continue
            for k in range(self.sub_size_list[i]):
                # 随机产生子图的大小
                sub_node_num = random.randint(3, self.hp.max_graph_size)
                seta = 5 * sub_node_num
                tem_vis = [0 for j in range(self.hp.node_num)]
                tem_node_set = set()
                sub_node_set = []
                tem_node_set.add(i)
                tem_vis[i] = 1
                # 限制子图大小
                # print(tem_node_set)
                while len(sub_node_set) < sub_node_num:
                    # print(sub_node_num)
                    choose_node = random.sample(tem_node_set, 1)
                    tem_node_set.remove(choose_node[0])
                    sub_node_set.append(choose_node[0])
                    # 限制筛选集合大小
                    if len(tem_node_set) < seta:
                        for j in self.G.neighbors(choose_node[0]):
                            if not tem_vis[j]:
                                tem_vis[j] = 1
                                tem_node_set.add(j)
                    if (len(tem_node_set) <= 0):
                        break
                self.subgraph_set.append(sub_node_set)