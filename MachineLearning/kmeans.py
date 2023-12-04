import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def knn_easy(cluster, dataset):
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)
    data_len = len(dataset)
    random_index = random.sample(range(0, data_len), cluster)
    miu_list = np.array([dataset[random_index[i]] for i in range(cluster)])
    cluster_type = np.array([0 for i in range(data_len)])
    times = 0
    while True:
        times += 1
        temp_miu_list = miu_list.copy()
        for i in range(cluster):
            if i == 0:
                distance = dataset - miu_list[i]
                distance = np.linalg.norm(distance, ord=2 ,axis=1)
            else:
                temp_distance = dataset - miu_list[i]
                temp_distance = np.linalg.norm(temp_distance, ord=2, axis=1)
                index = np.where((temp_distance < distance) == True)[0]
                for j in index:
                    distance[j] = temp_distance[j]
                    cluster_type[j] = i
                # print(cluster_type)
                # print(temp_distance < distance)
                # print(index)
        for i in range(cluster):
            index = (cluster_type == i)
            temp_data = dataset[index]
            data_mean = temp_data.mean(axis=0)
            temp_miu_list[i] = data_mean
            # print(temp_data.shape)
        if (temp_miu_list == miu_list).all():
            break
        else:
            miu_list = temp_miu_list
    print(times)
    return cluster_type, random_index
    # print(cluster_type)
                
            
if __name__ == '__main__':
    np.random.seed(1)
    # random.seed(1)
    data = np.random.randn(1000, 2)
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()
    
    k = 10
    test_data = [[x, y] for x, y in data]
    res_type, random_index = knn_easy(k, test_data)

    for i in range(k):
        plot_data = data[res_type == i]
        plt.scatter(plot_data[:, 0], plot_data[:, 1], label = i)
        
    for i in range(k):
        plt.scatter(data[i][0], data[i][1], s=50, c='blue', marker='x')
    
    plt.legend()
    plt.show()    

    # print(data)