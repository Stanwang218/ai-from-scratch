import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

# after embedding: batch_size, source_len, d_model

d_model = 512 # length of embedding
d_k = 8 # length of Q and K
d_v = 8 # length of V
d_ff = 2048 # length of feed forward network
heads = 6 # number of heads
batch_size = 100

class ScaledDot_Product(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q_s, k_s, v_s):
        dot_product = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d_k)
        # mask 
        dot_product = nn.Softmax(dim=-1)(dot_product)
        result = torch.matmul(dot_product, v_s)
        # (B, H, S, D(K))
        return result

class Multihead_Attention(nn.Module):
    def __init__(self) -> None:
        # input: batch_size(B), source_len(S), d_model(D)
        # Q, k, V
        super(Multihead_Attention, self).__init__()
        self.Q = nn.Linear(d_model, d_k * heads) # batch, source
        self.K = nn.Linear(d_model, d_k * heads)
        self.V = nn.Linear(d_model, d_v * heads)
        self.linear = nn.Linear(heads * d_k, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    # (B, S, D(model)) -project--> (B, S, D(k) * H) -split--> (B, S, H, D(K)) -transpose--> (B, H, S, D(K))
    # -Flatten--> (B, S, H * D(K)) -project--> (B, S, D(model))
    def forward(self, q_input, k_input, v_input):
        residue, batch_size = q_input, q_input.size(0)
        q_s = self.Q(q_input).view(batch_size, -1, heads, d_k).transpose(1, 2)
        k_s = self.K(k_input).view(batch_size, -1, heads, d_k).transpose(1, 2)
        v_s = self.V(v_input).view(batch_size, -1, heads, d_v).transpose(1, 2)
        result = ScaledDot_Product()(q_s, k_s, v_s)
        result = result.transpose(1, 2).reshape(batch_size, -1, heads * d_k)
        result = self.linear(result)
        return self.layer_norm(residue + result) 

class PointWise_FeedForward(nn.Module):
    # (B, S, D(model))
    def __init__(self):
        super(PointWise_FeedForward, self).__init__()
        self.mlp1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        residual = x
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        return nn.LayerNorm(d_model)(x + residual)
    
class Encoder_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead = Multihead_Attention()
        self.feedforward = PointWise_FeedForward()
    
    def forward(self, x):
        x = self.multihead(x, x, x)
        x = self.feedforward(x)
        return x

if __name__ == '__main__':
    layer = Encoder_Layer()
    t = torch.rand((2, 10, 512))
    # l = nn.Conv1d(512, 2048,1)
    summary(layer, (10,512))
    print(layer(t).shape)