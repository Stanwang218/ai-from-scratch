import time


def load_DataSet(path):
    with open(path,'r') as f:
        data = f.readlines()
        data_list = []
        print(len(data))
        for line in data:
            data_list.append(tuple(map(int,line[:-2].split(' '))))
        return data_list


def generate_l1(min_support):
    global dataset
    global length
    cnt = {}
    for tran in dataset:
        for item in tran:
            if cnt.get(item,0) == 0:
                cnt[item] = 1
            else:
                cnt[item] += 1

    l1 = []

    for item in cnt:
        if cnt[item] / length >= min_support:
            l1.append([item])

    # print(l1)
    # print(len(l1))
    return list(map(frozenset,l1))


def generate_frequent_set(candidate_set,min_support):
    global dataset
    global length
    global total
    _dataset = []
    cnt = {}
    # check every transaction in dataset
    for data in dataset:
        flag = True
        for candidate in candidate_set:
            if candidate.issubset(data):
                if flag:
                    _dataset.append(data)
                    flag = False
                # print(data)

                # calculate the number of the candidate
                if candidate not in cnt:
                    cnt[candidate] = 1
                else:
                    cnt[candidate] += 1
    l_k = []
    for key in cnt:
        if cnt[key] / total >= min_support:
            l_k.append(key)
    # print(_dataset)
    dataset = _dataset
    length = len(dataset)
    # print(length)
    return l_k


# obtain k-frequent item
def generate_candidate_set(ck_minus_one,k):
    ck = []

    # print(ck_minus_one)
    _length = len(ck_minus_one)
    for i in range(_length):
        for j in range(i+1,_length):
            list1 = list(ck_minus_one[i])[:k-1-1]
            list2 = list(ck_minus_one[j])[:k-1-1]
            # list1.sort()
            # list2.sort()
            if list1 == list2:
                ck.append(ck_minus_one[i] | ck_minus_one[j])
    return ck


def Apriori(min_support):
    global dataset

    # generate c1
    l1 = generate_l1(min_support)
    res = [l1]

    # the length of the candidate
    k = 2
    while len(res[k - 1 - 1]) >= 0:
        ck = generate_candidate_set(res[k - 1 - 1], k)
        lk = generate_frequent_set(ck,min_support)
        if len(lk) == 0:
            break
        else:
            res.append(lk)
            k += 1
    return res


if __name__ == '__main__':
    load_path = 'retail.dat'
    support = float(input())
    dataset = load_DataSet(load_path)
    total = len(dataset)
    # dataset = [(1,3,4),(2,3,5),(1,2,3,5),(2,5)]
    dataset = list(map(set,dataset))
    # print(dataset)
    length = len(dataset)
    # print(length)

    # print(generate_l1(support))

    start = time.clock()
    ans = Apriori(support)
    print(time.clock()-start)
    # print(ans)
    len_dict = {}
    lk = 1
    total_length = 0
    for k in ans:
        len_dict[lk] = len(k)
        total_length += len_dict[lk]
        lk += 1
    print("total length: {}".format(total_length))
    print(sorted(len_dict.items(), key=lambda x: x[0]))
    with open('Apriori.txt','w') as f:
        f.write(str(ans))
        f.close()