import os
import time
from tqdm import tqdm
import pandas as pd
import math


class Apriori():
    def create_L1(self, dataset, min_support, support_data):  # 遍历整个数据集生成c1候选集
        # c1候选集每个元素都是单个frozenset({i})
        item_count, appear = {}, {}   #appear 记录每个项出现的id
        L1 = set()
        for id, i in enumerate(dataset):  # 每个i就是一个处方
            for j in i:  # 便利处方的每一种药
                item = frozenset([j])
                item_count[item] = item_count.get(item, 0) + 1
                if item not in appear:
                    appear[item] = [id]
                else:
                    appear[item].append(id)


        for item in item_count:  # 将满足支持度的候选项添加到频繁项集中
            if item_count[item] >= min_support:
                L1.add(item)
                support_data[item] = item_count[item]

        return L1, appear



    def create_Lk(self, Lk_1, size, min_support, appear):  # 通过频繁项集Lk-1创建ck候选项集
        L2 = set()
        l = len(Lk_1)
        lk_list = list(Lk_1)
        new_appear = {}
        for i in tqdm(range(l)):
            for j in range(i + 1, l):  # 两次遍历Lk-1，找出前n-1个元素相同的项
                Ck_item = lk_list[i] | lk_list[j]
                if len(Ck_item) == size:  # 只有最后一项不同时，生成下一候选项
                    a = set(appear[lk_list[i]])
                    b = set(appear[lk_list[j]])
                    if len(a & b) >= min_support:
                        L2.add(Ck_item)
                        new_appear[Ck_item] = list(a&b)

        return L2,new_appear


    def generate_L(self, data_set, min_support):  # 用于生成所有频繁项集的主函数，k为最大频繁项的大小

        support_data = {}  # 用于保存各频繁项的支持度
        L1, appear = self.create_L1(data_set, min_support, support_data)  # 生成C1

        # L2 = self.generate_lk_by_ck(data_set, C2, min_support, support_data)

        Lksub1 = L1.copy()  # 初始时Lk-1=L1

        L = []
        L.append(L1)

        i = 2
        while (True):
            # Ci = self.create_ck(Lksub1, i)  # 根据Lk-1生成Ck
            # Li = self.generate_lk_by_ck(data_set, Ci, min_support, support_data)  # 根据Ck生成Lk
            Li,appear = self.create_Lk(Lksub1, i, min_support, appear)
            if len(Li) == 0: break
            Lksub1 = Li.copy()  # 下次迭代时Lk-1=Lk
            L.append(Lksub1)
            i += 1
        sum = 0
        for i in range(len(L)):
            print("frequent item {}：{}".format(i + 1, len(L[i])))
            sum += len(L[i])
        print("频繁项集的总数为", sum)
        # i+1项集有几个频繁项
        return L, support_data

    def generate_R(self, dataset, min_support, min_conf):
        L, support_data = self.generate_L(dataset, min_support)  # 根据频繁项集和支持度生成关联规则
        rule_list = []  # 保存满足置信度的规则
        sub_set_list = []  # 该数组保存检查过的频繁项
        for i in range(0, len(L)):
            for freq_set in L[i]:  # 遍历Lk
                for sub_set in sub_set_list:  # sub_set_list中保存的是L1到Lk-1
                    if sub_set.issubset(freq_set):  # 检查sub_set是否是freq_set的子集
                        # 检查置信度是否满足要求，是则添加到规则
                        conf = support_data[freq_set] / support_data[freq_set - sub_set]
                        big_rule = (freq_set - sub_set, sub_set, conf)
                        if conf >= min_conf and big_rule not in rule_list:
                            rule_list.append(big_rule)
                sub_set_list.append(freq_set)
        rule_list = sorted(rule_list, key=lambda x: (x[2]), reverse=True)
        return rule_list


if __name__ == "__main__":
    t1 = time.time()

    data = pd.read_csv("retail.dat", header=None, names=['thing'])
    data = [sorted(list(set(map(int, i[0].split(" ")[:-1])))) for i in data.values]
    apriori = Apriori()
    # rule_list = apriori.generate_R(data, min_support=math.floor(88162*0.02), min_conf=0.7)

    L, support_data = apriori.generate_L(data, math.ceil(88162 * 0.007))

    print(time.time() - t1)
