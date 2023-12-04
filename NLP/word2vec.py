import torch
import torch.nn as nn
import os
import sys
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

os.chdir(sys.path[0])

class word2vec(nn.Module):
    def __init__(self, voc_size, d = 2, window_size = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(voc_size, d)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(d * window_size, voc_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.output(x)
        return(x)
    
class word_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        d1, d2, s, x, y = read_data()
        self.word_dict = d1
        self.index_dict = d2
        self.voc_size = s
        self.data = x
        self.target = y
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]

def read_data():
    x, y = [], []
    with open("./text1.txt") as f:
        data = f.readlines()[0].lower()
        sentence_seq = data.split('.')[:-1]
        sentences = " ".join(sentence_seq)
        word_list = list(set(sentences.split()))
        word_dict = {u:i for i,u in enumerate(word_list)}
        index_dict = {word_dict[key]:key for key in word_dict.keys()}
        voc_size = len(word_list)
        skip_gram = []
        for seq in sentence_seq:
            word_seq = seq.split()
            # window size equals to 1
            for i in range(1, len(word_seq) - 1):
                target = word_dict[word_seq[i]]
                context = [word_dict[word_seq[i-1]], word_dict[word_seq[i+1]]]
                x.append(context)
                y.append(target)
        return word_dict, index_dict, voc_size, np.array(x), np.array(y)

def train_model():

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=10e-5)
    
    batch_size = 100
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    loss_value = []
    print(dataset.voc_size)
    for epoch in range(2000):
        optimizer.zero_grad()
        for x, y in dataloader:
            output = model(x)
            # print(output.shape)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loss_value.append(loss.item())
            # print(loss.item())
            # break
        # break
    
    torch.save(model.state_dict(),"./model/word2vec.pth")
    
    # print(loss_value)
    # plt.ylim([0, 10])
    # plt.plot(loss_value)
    # plt.show()

def test_model():
    model = word2vec(dataset.voc_size)
    model.load_state_dict(torch.load('./model/word2vec.pth'))
    test = ["cities", "a"]
    model.eval()
    index = [dataset.word_dict[i] for i in test]
    index = torch.LongTensor(index).reshape(1,-1)
    result = torch.argmax(model(index)).item()
    print(dataset.index_dict[result])
    
def load_weight():
    model = word2vec(dataset.voc_size)
    model.load_state_dict(torch.load('./model/word2vec.pth'))
    weight = list(model.parameters())[0].detach().numpy()
    print(weight[0])
    for key in dataset.word_dict:
        index = dataset.word_dict[key]
        x, y = weight[index][0], weight[index][1]
        plt.scatter(x,y)
        plt.annotate(key, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()        
    
    # print(weight.shape)
    


if __name__ == '__main__':
    dataset = word_dataset()
    model = word2vec(dataset.voc_size)
    train_model()
    test_model()
    # train_model()
    load_weight()