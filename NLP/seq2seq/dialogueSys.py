import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
import sys
import json
import math
import matplotlib.pyplot as plt
from rnn import *
import numpy as np
import seaborn as sns

os.chdir(sys.path[0])

def sentence_process(sentences):
    s = set()
    max_len = 0
    for sent in sentences:
        words = sent[:-1].split()
        max_len = max(max_len, len(words))
        s.update(words)
    return s, max_len

def save_json(d : dict, name : str):
    jsonfile = json.dumps(d)
    with open(f'{name}.json', 'w') as f:
        f.write(jsonfile)
    f.close()

def preprocess_corp():
    question_sentences = open('./question.txt').readlines()
    answer_sentences = open('./answer.txt').readlines()
    quest_set, max_quest = sentence_process(question_sentences)
    answer_set, max_ans = sentence_process(answer_sentences)
    wordbag = quest_set.union(answer_set)
    word2num, num2word = dict(), dict()
    special_symbol = ['<PAD>', '<START>', '<END>']
    for i in range(len(special_symbol)):
        word2num[special_symbol[i]] = i
        num2word[i] = special_symbol[i]

    num = len(special_symbol)
    i = 0
    for element in wordbag:
        if element in special_symbol:
            continue
        word2num[element] = num + i
        num2word[num + i] = element
        i += 1
        
    word2num['total length'] = max(max_ans, max_quest)
    save_json(word2num, 'word2num')
    save_json(num2word, 'num2word')

class BagofWords(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.question_sentences = open('./question.txt').readlines()
        self.answer_sentences = open('./answer.txt').readlines()
        self.word2num, self.num2word = json.load(open('./word2num.json', 'r')), json.load(open('./num2word.json', 'r'))
        self.wordnum = len(self.word2num)
        self.max_len = self.word2num['total length']
        self.corpus = []
        self.quest_len = []
        for quest, ans in zip(self.question_sentences, self.answer_sentences):
            quest = quest.split()[1: ]
            ans = ans.split()[1 :]
            quest_index, ans_index = np.zeros(self.max_len), np.zeros(self.max_len)
            quest_len, ans_len = len(quest), len(ans)
            quest = [self.word2num[i] for i in quest]
            ans = [self.word2num[i] for i in ans]
            quest_index[: quest_len] = quest
            ans_index[: ans_len] = ans
            self.quest_len.append(quest_len)
            self.corpus.append((quest_index, ans_index))
            
            
    def __getitem__(self, index):
        quest, ans = self.corpus[index]
        return torch.tensor(quest, dtype=torch.long), torch.tensor(ans, dtype=torch.long), self.quest_len[index]
    
    def __len__(self):
        return len(self.corpus)
        
    @staticmethod
    def sentence_process(sentences):
        s = set()
        for sent in sentences:
            words = sent[:-1].split()

            s.update(words)
        return s
    
    @staticmethod
    def save_json(d : dict, name : str):
        jsonfile = json.dumps(d)
        with open(f'{name}.json', 'w') as f:
            f.write(jsonfile)
        f.close()

def create_mask(length_list, max_len):
    l = length_list.shape[0]
    mask = torch.zeros([l, max_len])
    for i, pos in enumerate(length_list.numpy()):
        mask[i, pos:] = 1
    return mask

def train(pretrained = False):
    device = torch.device("cuda:2")
    word_set = BagofWords()
    print(word_set.wordnum)
    model = dialogNN(word_set.wordnum, 32, 32, 3).to(device)
    # model = myNeuralNetwork(num_word=word_set.wordnum, max_len=word_set.max_len).to(device)
    if pretrained:
        model.load_state_dict(torch.load('./model.pth'))
    # print(word_set.max_len)
    loss_list = []
    epochs = 2000
    optimizer = optim.SGD(model.parameters(), lr=1)
    batch_size = 1024
    dataloader = DataLoader(dataset=word_set, batch_size=batch_size)
    for epoch in range(epochs):
        total_loss = 0
        for x, y, x_length in dataloader:
            x = x.to(device)
            y = y.to(device)
            decoder_input = torch.LongTensor([[0 for _ in range(x.shape[0])]]).view(-1, 1).to(device)
            mask = create_mask(x_length, word_set.max_len).to(device)
            optimizer.zero_grad()
            loss = 0
            for t in range(word_set.max_len):
                temp_mask = (mask[: , t] != 1)
                output = model(x, decoder_input, mask) # B, num_words
                selected_output = output[temp_mask]
                selected_y = y[temp_mask][:, t]
                if selected_y.shape[0] == 0:
                    break
                loss += torch.nn.functional.cross_entropy(selected_output, selected_y)
                if (epoch + 1) % 100 == t < 10:
                    print([word_set.num2word[str(i)] for i in torch.argmax(output, dim = 1).detach().cpu().numpy()[:10]])
                    print([word_set.num2word[str(i)] for i in y[:, t].squeeze().detach().cpu().numpy()[:10]])                    
                decoder_input = y[:, t].unsqueeze(1)
            loss = loss / word_set.max_len
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss = total_loss / (len(dataloader))
        print(f"{epoch + 1} : {total_loss}")
        loss_list.append(total_loss)
    torch.save(model.state_dict(), './model.pth')
    plt.plot(loss_list)
    plt.savefig('./loss.png')
    
def test():
    wordset = BagofWords()
    model = dialogNN(wordset.wordnum, 32, 32, 3)
    model.load_state_dict(torch.load('./model.pth', map_location='cpu'))
    model.eval()
    test_sentences = "What time does the next tram to the park depart <END>"
    test_sentence = test_sentences.split()
    origin_sentence = test_sentence.copy()
    test_len = len(test_sentence)
    start_sentence = ["<START>"]
    mask = torch.ones(1, wordset.max_len)
    mask[0, : test_len] = 0
    start_sentence.extend(["<PAD>" for i in range(wordset.max_len - 1)])
    test_sentence.extend(["<PAD>" for i in range(wordset.max_len - test_len)])
    word2num, num2word = json.load(open('./word2num.json')), json.load(open('./num2word.json'))
    
    # for i in range(len(test_sentence)):
    #     test_sentence[i] = word2num[test_sentence[i]]
    for i in range(wordset.max_len):
        start_sentence[i] = word2num[start_sentence[i]]
        test_sentence[i] = word2num[test_sentence[i]]
    src_tensor = torch.tensor(test_sentence, dtype=torch.long).view(1, -1)
    print(test_sentences)
    attn_list, next_word_list = [], []
    for i in range(wordset.max_len - 1):
        decoder_input = torch.tensor(start_sentence[i]).view(1, -1)
        output, attn = model(src_tensor, decoder_input, mask)
        word_index = torch.argmax(output).item()
        next_word = num2word[str(word_index)]
        if next_word == "<END>":
            break
        print(next_word, end=" ")
        start_sentence[i + 1] = word_index
        attn_list.append(attn.squeeze(0).detach().numpy())
        next_word_list.append(next_word)
    # print(attn_list)
    data = np.concatenate(attn_list, axis=0)
    truncate_num = 6
    plt.figure(figsize=(10, 6))
    sns.heatmap(data[:truncate_num, :test_len], cmap='Blues' ,xticklabels=origin_sentence[: test_len],yticklabels=next_word_list[: truncate_num], annot=True, fmt=".2f")
    plt.title('Attention Heatmap for \"What time does the next tram to the park depart\"')
    plt.xlabel('Input Sequence')
    plt.ylabel('Output Sequence')
    
    # show graphics
    plt.show()
   
# preprocess_corp()
# train(pretrained=False)
test()
# word_set = BagofWords()
# print(word_set[1])
# print(word_set.max_len)
# print(len(word_set.wordbag))
    