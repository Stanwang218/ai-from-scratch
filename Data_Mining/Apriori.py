import time


def loadDataSet(path):
    with open(path,'r') as f:
        data = f.readlines()
        data_list = []
        print(len(data))
        num = 0
        for line in data:
            data_list.append(list(map(int,line[:-2].split(' '))))
        return data_list


def createC1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def ScanD(dataset, c, min_support):
    cnt = {}
    for data in dataset:
        for x in c:
            if x.issubset(data):
                if x not in cnt:
                    cnt[x] = 1
                else:
                    cnt[x] += 1
    # calculate the times of one number
    length = len(dataset)
    rest_list = []
    support_data = {}
    for key in cnt:
        if cnt[key] / length >= min_support:
            rest_list.append(key)
        support_data[key] = cnt[key] / length
    # print(support_data)

    return rest_list,support_data


def genertateSet(ck,k):
    # the length of the value in ck is k-1
    # the aim is to generate the k_length set
    retlist = []
    length = len(ck)
    for i in range(length):
        for j in range(i+1,length):
            list1 = list(ck[i])[:k-2]
            list2 = list(ck[j])[:k-2]
            list1.sort()
            list2.sort()
            if list1 == list2:
                retlist.append(ck[i]|ck[j])
    return retlist


def Apriori(dataset,minsupport):
    C1 = createC1(dataset)
    # 1,2,3,4,5
    D = list(map(set,dataset))
    L1,support_data = ScanD(D,C1,minsupport)
    L = [L1]
    # answer L1 is the one-length set that satisfied the minimum support
    k = 2
    while len(L[k-2]) > 0:
        Ck = genertateSet(L[k-2],k)
        Lk,supK = ScanD(D,Ck,minsupport)
        support_data.update(supK)
        if len(Lk) == 0:
            break
        L.append(Lk)
        k += 1
    return L,support_data


if __name__ == '__main__':
    load_path = 'retail.dat'
    # loadDataSet(load_path)
    start = time.clock()
    L,support_data = Apriori(loadDataSet(load_path),0.02)
    print(L)
    _len = 0
    for x in L:
        _len += len(x)
    print(support_data)
    print(time.clock()-start)
    print(_len)