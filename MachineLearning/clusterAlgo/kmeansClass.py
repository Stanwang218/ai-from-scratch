import numpy as np
import pandas as pd
import sys
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

os.chdir(sys.path[0])

class MyKmeans:
    def __init__(self, k = 8, max_iter = 100):
        self.k = k
        self.max_iter = max_iter
        self.centroid = None
        self.labels = None
        
    def fit(self, x:np.ndarray, y:np.ndarray):
        if not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            x = np.array(x)
            y = np.array(y)
        self.x, self.y = x, y
        data_len = len(self.y)
        random_index = np.random.choice([i for i in range(data_len)], size=self.k, replace = False)
        mean_vector = np.array([self.x[i] for i in random_index]) # k x dim
        c = np.zeros(data_len)
        c_label = np.zeros(self.k)

        for times in range(self.max_iter):
            old_mean_vector = mean_vector.copy()
            cluster_size = np.zeros(self.k)
            for i in range(data_len):
                # print(i)
                dist_vec = np.linalg.norm(self.x[i] - mean_vector, ord=2, axis=1)
                cluster = np.argmin(dist_vec)
                c[i] = cluster
                cluster_size[cluster] += 1
            
            for i in range(self.k):
                mean_vector[i] = np.sum(self.x[c == i], axis=0)
                mean_vector[i] = mean_vector[i] / cluster_size[i]
            
            if np.array_equal(mean_vector, old_mean_vector):
                break
        
        for i in range(self.k): # find the label for each cluster
            elements, cnts = np.unique(self.y[c == i], return_counts=True)
            c_label[i] = elements[np.argmax(cnts)]
        
        self.centroid, self.labels = mean_vector, c_label
    
    def predict(self, x_predict):
        data_len = len(x_predict)
        res = np.zeros(data_len)
        for i in range(data_len):
            dist_vec = np.linalg.norm(x_predict[i] - self.centroid, ord=2, axis=1)
            res[i] = self.labels[np.argmin(dist_vec)]
        
        return res
    
    def score(self, y_pred, y_true):
        data_len = len(y_true)
        return np.sum(y_pred == y_true) / data_len * 100
            
def prepare_data(fract = 0.2):
    path = "./Cities"
    city_file = os.listdir(path)
    train_files = city_file[:2]
    test_files = city_file[2:]
    train_data, test_data = [],[]
    temp_data = [[], [], [], []]
    val_data = []
    for file in train_files:
        data = pd.read_csv(f"{path}/{file}")
        temp_val_data = data.sample(frac=0.2)
        temp_train_data = data.drop(temp_val_data.index)
        col = data.columns
        x_train, y_train = temp_train_data[col[:-1]].values, temp_train_data[col[-1]].values
        x_val, y_val = temp_val_data[col[:-1]].values, temp_val_data[col[-1]].values
        temp_data[0].append(x_train)
        temp_data[1].append(y_train)
        temp_data[2].append(x_val)
        temp_data[3].append(y_val)
    
    train_data.append(np.concatenate([temp_data[0][0], temp_data[0][1]], axis=0))
    train_data.append(np.concatenate([temp_data[1][0], temp_data[1][1]], axis=0))
    val_data.append(np.concatenate([temp_data[2][0], temp_data[2][1]], axis=0))
    val_data.append(np.concatenate([temp_data[3][0], temp_data[3][1]], axis=0))
    
    temp_data = [[], []]
    for file in test_files:
        data = pd.read_csv(f"{path}/{file}")
        col = data.columns
        x, y = data[col[:-1]].values, data[col[-1]].values
        temp_data[0].append(x)
        temp_data[1].append(y)
    
    return train_data[0], train_data[1], val_data[0], val_data[1], temp_data # x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == '__main__':
    
    x_train, y_train, x_val, y_val, test_data = prepare_data()
    val_score, train_score, gz_score, sh_score = [], [], [], []
    clusters = 50
    repeat = 20
    result = pd.DataFrame(columns = ["train","val","Guangzhou","Shanghai"], index = range(1, clusters+1))
    
    for i in range(clusters):
        n_cluster = i + 1
        print(f"cluster: {n_cluster} ..")
        model = MyKmeans(k = n_cluster)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        train_score.append(model.score(train_pred, y_train))
        val_score.append(model.score(val_pred, y_val))
        gz_temp_score, sh_temp_score = 0, 0
        for time in range(repeat):
            for j in range(2):
                x_test, y_test = test_data[0][j], test_data[1][j]
                y_pred = model.predict(x_test)
                if j == 0:
                    gz_temp_score += model.score(y_pred, y_test)
                else:
                    sh_temp_score += model.score(y_pred, y_test)
        gz_score.append(gz_temp_score / repeat)
        sh_score.append(sh_temp_score / repeat)
        result.loc[n_cluster] = [train_score[i], val_score[i], gz_score[i], sh_score[i]]
        

    scores = [train_score, val_score, gz_score, sh_score]
    for item in result.columns:
        plt.plot(result.index, result[item], label=item)
    plt.title("Accuracy VS number of cluster")
    plt.legend()
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.xticks(range(min(result.index), max(result.index)+1))

    plt.show()

        
    
    # model = KMeans(n_cluster)
    # model.fit(x_train, y_train)
    # labels = model.labels_
    # pred_cluster = model.predict(x_test)
    # cluster_label = np.zeros(n_cluster)
    # for i in range(n_cluster):
    #     cluster_index = (labels == i)
    #     label_cluster = y_train[cluster_index]
    #     ele, cnts = np.unique(label_cluster, return_counts=True)
    #     cluster_label[i] =  ele[np.argmax(cnts)]
        
    # test_label = np.array([cluster_label[i] for i in pred_cluster])

    
    # score = model.score(x_test, x_train)
    # print(np.sum(test_label == y_test) / len(y_test) * 100)
        
    # y_pred = model.predict(x_test)