from multiprocessing import Process
import random
import numpy as np
class spliter (Process):
    def __init__(self, threadID, args, subgraph_set, Go, Gn):
        Process.__init__(self)
        self.threadID = threadID
        self.hp = args
        self.subgraph_set = subgraph_set
        self.Go = Go
        self.Gn = Gn
        self.pro = []
        self.output = []
    def run(self):
        for sub_set in self.subgraph_set:
            tem_pro = []
            tem_out = []
            for i, node_1 in enumerate(sub_set):
                pro_vec = []
                for j, node_2 in enumerate(sub_set):
                    if node_1 not in self.Go.nodes():
                        continue
                    if node_2 in self.Go.neighbors(node_1):
                        # print(j," ",rem2origin[node_1]," ",rem2origin[node_2])
                        pro_vec.append(node_2)
                        # pro_vec[j] = 1.0

                tem_pro.append(pro_vec)
            for i, node_1 in enumerate(sub_set):
                # 为了输入输出需要，固定成和embdeding一样的维度
                pro_out = np.zeros((len(sub_set)))
                for j, node_2 in enumerate(sub_set):
                    if node_1 not in self.Gn.nodes():
                        continue
                    if node_2 in self.Gn.neighbors(node_1):
                        pro_out[j] = 1.0
                # 对标签输出进行平滑处理
                # pro_out = label_smoothing(pro_out)
                # 对标签输出进行正则化处理
                # pro_vec = preprocessing.normalize(pro_vec, norm='l2')
                tem_out.append(pro_out)
            self.pro.append(tem_pro)
            self.output.append(tem_out)