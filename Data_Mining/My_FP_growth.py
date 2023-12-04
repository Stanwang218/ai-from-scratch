from itertools import combinations
import time


def test_dataset():
    return [['f', 'a', 'c', 'd', 'g', 'i', 'm', 'p'], ['a', 'b', 'c', 'f', 'l', 'm', 'o'],
            ['b', 'f', 'h', 'j', 'o'], ['b', 'c', 'k', 's', 'p'], ['a', 'f', 'c', 'e', 'l', 'p', 'm', 'n']]


def load_DataSet(path):
    with open(path, 'r') as f:
        data = f.readlines()
        data_list = []
        print(len(data))
        for line in data:
            data_list.append(tuple(map(int, line[:-2].split(' '))))
        return data_list


class element:
    def __init__(self, item, frequency):
        self.item = item
        self.frequency = frequency
        self.head = None
        self.now = None

    def __lt__(self, other):
        if self.frequency == other.frequency:
            return self.item < other.item
        else:
            return self.frequency < other.frequency


class node:
    def __init__(self, item, num=1):
        self.item = item
        self.num = num
        self.brother = None
        self.next = []
        self.pre = []

    def __eq__(self, other):
        return self.item == other.item

    def __hash__(self):
        return hash(self.item)


def create_limb(head_table, head, temp_list, index,single=True):
    # head table refers to header table
    # head refers to root
    # temp_list refers to the simplified transaction
    # index refers to the present index
    if index == len(temp_list):
        # print(single)
        if single:
            return True
        else:
            return False

    number = temp_list[index]
    # present node
    temp_node = node(number)
    # print(temp_node.next)
    if temp_node not in head.next:
        # first time to be created in the tree
        if head_table[number].head is None:
            head_table[number].head = temp_node
            head_table[number].now = temp_node
        else:
            head_table[number].now.brother = temp_node
            head_table[number].now = temp_node
        head.next.append(temp_node)
        if len(head.next) >= 2:
            single = False
        temp_node.pre.append(head)
        return create_limb(head_table, temp_node, temp_list, index + 1,single)
    else:
        # if the node has appeared in the header table
        _index = head.next.index(temp_node)
        # to find the index of the node in the head.next
        head.next[_index].num += 1
        return create_limb(head_table, head.next[_index], temp_list, index + 1,single)


def generate_table(min_support,data):
    # length = len(data)
    global total
    cnt = {}
    for tran in data:
        for item in tran:
            if cnt.get(item) is None:
                cnt[item] = 1
            else:
                cnt[item] += 1
    # print(cnt)
    # l1 = []
    element_list = []
    # print(cnt)
    # create table
    for item in cnt:
        if cnt[item] / total >= min_support:
            element_list.append(element(item, cnt[item]))

    return element_list


def create_FP_tree(data,min_support,head=None):
    num_list = []
    ans = []
    dicts = {}
    # a dictionary item : element

    head_table = generate_table(min_support,data)
    head_table.sort(reverse=True)
    # head_table = [element('f',4),element('c',4),element('a',3),element('b',3),element('m',3),element('p',3)]
    for x in head_table:
        num_list.append([x.item])
        dicts[x.item] = x

    # [frozenset[]]

    # x : header table

    # 1 frequent item
    if head is None:
        maps = list(map(frozenset,num_list))
        ans.extend(maps)

    # print(head_table)
    # num_list = ['f','c','a','b','m','p']
    # print(dicts)
    # print(num_list)

    single = True
    root = node(head)
    # print(num_list)

    # create the FP_tree
    for transaction in data:
        temp_list = []
        for num in num_list:
            if num[0] in transaction:
                temp_list.append(num[0])
            # simplify the transaction

        # print(temp_list)
        # for each simplified transaction to create FP_tree
        if temp_list:
            # print(temp_list)
            # print(create_limb(dicts, root, temp_list, 0))
            if not create_limb(dicts, root, temp_list, 0):
                single = False

    # check the link
    # print(root.next)
    # for key in dicts:
    #     temp_node = dicts[key].head
    #     while temp_node:
    #         print(temp_node.num)
    #         temp_node = temp_node.brother
    #     print("\n")

    # print(root)

    if single:
        ptr = root.next
        item_list = []
        # single root, append the list from the root from top to bottom
        while ptr:
            item_list.append(ptr[0].item)
            ptr = ptr[0].next
        combination_list = []
        length = len(item_list)
        # generate combination
        for i in range(1, length + 1):
            combination_list.extend(list(map(set, list(combinations(item_list, i)))))
        # combination_list.extend([set()])
        for item in combination_list:
            item.add(root.item)
        # print(combination_list)
        return combination_list
        # create the set
    else:
        # ans = []
        # if head is None:
        #     ans.extend(list(map(set,num_list)))

        base = FP_growth_base(dicts)
        # generate the frequent pattern base(dictionary)

        # check the base
        for key in base:
            if len(base[key]) != 0:
                data_list = []
                for tran in base[key]:
                    # print(tran)
                    for times in range(base[key][tran]):
                        data_list.append(list(tran))

                # print(data_list)
                ans.extend(create_FP_tree(data_list,min_support,key))
                # FP_growth(min_support,data_list)
        # print(base)
        return ans
        # print(root.next[0].num)


def FP_growth_base(head_table):
    ans_base = {}
    # go through the head table
    for row in head_table:
        # row is a string
        dicts = {}
        nodes = head_table[row].head
        # the pointer

        while True:
            temp = nodes.pre[0]
            temp_order = []
            # print(temp.item)

            # search for the previous node
            while True:
                # print(temp.pre)
                # print(temp.item)
                if temp.item is None:
                    break
                temp_order.append(temp.item)
                if not temp.pre:
                    break
                temp = temp.pre[0]
            if temp_order:
                # temp_order.reverse()
                temp_order = frozenset(temp_order)
                dicts[temp_order] = nodes.num
            nodes = nodes.brother
            if nodes is None:
                break

            # for each dicts to create fp_tree
        # print(dicts)
        ans_base[row] = dicts
    return ans_base
    # print(ans_base)


def FP_growth(min_support,data):
    return create_FP_tree(data,min_support)


if __name__ == '__main__':
    sup = float(input())
    load_path = 'retail.dat'
    dataset = load_DataSet(load_path)
    # dataset = test_dataset()
    total = len(dataset)
    start = time.clock()
    ans = FP_growth(sup,dataset)
    print(time.clock() - start)
    ans = list(set(map(frozenset,ans)))
    len_dict = {}
    # l1 = []
    for item in ans:
        length = len(item)
        len_dict[length] = len_dict.get(length,0) + 1
    total_length = 0
    for key in len_dict:
        total_length += len_dict[key]
    # print(l1)
    print("total length: {}".format(total_length))
    print(sorted(len_dict.items(),key=lambda x:x[0]))
    with open('FP_growth_res.txt','w') as f:
        f.write(str(ans))
        f.close()
    # print(len(ans))
    # print(ans)
    # print(FP_growth(0.5,dataset))
    # FP_growth_base(root,dicts)
    # for x in l1:
    #     print("{} : {}".format(x.item,x.frequency))

    # tree = FP_tree()
    # print(l1)
