from baseCluster import basecluster
from sklearn.cluster import dbscan
import numpy as np
import matplotlib.pyplot as plt

class DBSCAN(basecluster):
    def __init__(self, eps = 1, minPts = 5) -> None:
        super().__init__()
        self.queue = []
        self.eps = eps
        self.minPts = minPts
        self.core_set = []
        
    
    def fit(self, x, y = None):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        data_len = len(x)
        d = dict()
        for i in range(data_len):
            tempx = x - x[i]
            dist = np.linalg.norm(tempx, ord=2, axis=1)
            index = np.where(dist <=  self.eps)[0]
            if len(index) >= self.minPts:
                self.core_set.append(i)
                d[i] = index
        if len(self.core_set) == 0:
            raise ValueError; "There is no core object"
        k = 0 # number of cluster
        
        core_set = set(self.core_set)
        cluster_res = [-1 for i in range(data_len)] # return ans
        
        while len(core_set) > 0:
            # choose one core point
            self.queue.append(self.core_set[0]) # add into queue
            
            while len(self.queue) != 0:
                
                temp_point = self.queue.pop(0)
                cluster_res[temp_point] = k
            
                if temp_point in self.core_set:
                    core_set.remove(temp_point)
                    for i in d[temp_point]:
                        if i == temp_point:
                            continue
                        if cluster_res[i] == -1:
                            self.queue.append(i)

                self.core_set = list(core_set)

            k += 1
        return cluster_res
            
    
    def pred(self):
        return super().pred()
    
    def score(self):
        return super().score()
    
def resultCompare(eps_, minPts_):
    model = DBSCAN(eps=eps_, minPts=minPts_)
    np.random.seed(1)
    x = np.random.normal(size=(100, 2))
    ans = model.fit(x)
    num_type = np.unique(ans)
    plt.subplot(211)
    for i in num_type:
        selected_index = (ans == i)
        plt.scatter(x[selected_index][:, 0], x[selected_index][:, 1], label = f"{i}")
    plt.legend()
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    
    
    core_samples, labels = dbscan(x, eps = eps_, min_samples=minPts_)
    
    num_type = np.unique(labels)
    print(num_type)

    plt.subplot(212)
    for i in num_type:
        selected_index = (labels == i)
        print(len(x[selected_index]))
        plt.scatter(x[selected_index][:, 0], x[selected_index][:, 1], label = f"{i}")
    plt.legend()
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.show()
    

if __name__ == '__main__':
    resultCompare(0.3, 5)