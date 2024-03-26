import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import json
os.chdir(sys.path[0])

class encoder(nn.Module):
    def __init__(self, input_sz = 32, hidden_sz = 32, num_layers = 3) -> None:
        super().__init__()
        # input vector is B, S, D
        self.hidden_sz = hidden_sz
        self.numLayers = num_layers
        self.gru = nn.GRU(input_sz, hidden_sz, num_layers, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        encoder_output, hidden_output = self.gru(x)
        encoder_output = encoder_output[:, :, : self.hidden_sz] + encoder_output[:, :, self.hidden_sz: ]
        hidden_output = hidden_output[: self.numLayers]
        return encoder_output, hidden_output
    

class decoder(nn.Module):
    def __init__(self, input_sz = 32, hidden_sz = 32, num_layers = 3, output_sz = 200) -> None:
        super().__init__()
        self.hidden_sz = hidden_sz
        self.numLayers = num_layers
        self.gru = nn.GRU(input_sz, hidden_sz, num_layers, batch_first=True)
        self.concat = nn.Linear(self.hidden_sz * 2, self.hidden_sz) # concat context and encoder output
        self.classify = nn.Linear(self.hidden_sz, output_sz)
        
    def forward(self, decoder_input, encoder_output, last_hidden, mask = None):
        """input shape

        Args:
            decoder_input (tensor): B, 1, input_sz
            encoder_output (tensor): B, S, input_sz
            last_hidden (tensor): num_layer, B ,input_sz
        """
        assert decoder_input.shape[1] == 1; f"The input should be one word"
        decoder_output, decoder_hidden = self.gru(decoder_input, last_hidden) # B, 1, input_sz
        attn_scores = decoder_output.bmm(encoder_output.transpose(1, 2)) # B, 1, input_sz
        if mask is not None:
            mask = (mask == 1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill_(mask, -1e9)
        # mask
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=2)
        context = attn_scores.bmm(encoder_output).squeeze(1)
        decoder_output = decoder_output.squeeze(1)
        concat_tensor = torch.cat([decoder_output, context], dim=1)
        out = self.concat(concat_tensor)
        out = self.classify(out)
        return out, attn_scores
    
class dialogNN(nn.Module):
    def __init__(self, num_words, input_sz, hidden_sz, num_layers = 3):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_words, input_sz)
        self.encoder = encoder(input_sz, hidden_sz, num_layers)
        self.decoder = decoder(input_sz, hidden_sz, num_layers, num_words)
        
    def forward(self, x, y, mask):
        x = self.embedding(x)
        y = self.embedding(y)
        encoder_output, encoder_hidden = self.encoder(x)
        out, attn_scores = self.decoder(y, encoder_output, encoder_hidden, mask)
        # print(attn_scores.detach().numpy())
        return out, attn_scores

def vec_dist(word_dict, num_dict):
    # afternoon
    index = word_dict["afternoon"]
    model = dialogNN(1281, 32, 32, 3)
    model.load_state_dict(torch.load('./model.pth', map_location='cpu'))
    embedding_weight = list(model.parameters())[0]
    i_vector = embedding_weight[index].view(1, -1)
    norm_dist = torch.norm(i_vector-embedding_weight, dim=1)
    values, indices = torch.topk(norm_dist, k=10, largest=False)
    indices = indices.numpy()
    labels=[num_dict[str(i)] for i in indices]
    print([num_dict[str(i)] for i in indices])
    print(values.data)
    plt.bar(labels, values.data)
    # plt.xticks(range(0,10), labels=labels)
    plt.show()
    
if __name__ == '__main__':
    word2num = json.load(open('./word2num.json'))
    num2word = json.load(open('./num2word.json'))
    vec_dist(word2num, num2word)
    # print(torch.topk(), k=10)
    # for name, para in model.named_parameters():
    #     print(name)