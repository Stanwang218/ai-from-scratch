import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, x, eps = 1e-6):
        super().__init__()
        # nn.Parameter will add tensors into the module.parameters()
        # beta * (x - miu) + gamma, for each feature
        self.beta = nn.Parameter(torch.zeros(x))
        self.gamma = nn.Parameter(torch.ones(x))
        self.eps = eps
    
    def forward(self, x):
        """
        x: B, L (number of Q), F (feature)
        """
        x_mean = torch.mean(x, dim=-1, keepdim=True) # B, L
        x_std = torch.std(x, dim=-1, keepdim=True) # B, L
        return self.ones_parameter * (x - x_mean) / (x_std + self.eps)

class ScaledDotProduct(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, q, k, v):
        d_k = q.size(-1)
        # B, H_num, L, H_dim
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)
        scores = f.softmax(scores, dim=-1)
        # B, H_num, L, d_v
        return torch.matmul(scores, v)
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super().__init__()
        head_dim = d_model // num_heads
        assert head_dim * num_heads == d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.attention_function = ScaledDotProduct()
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v):
        batsz = q.size(0)
        # B, SouceLength, d_model -> B, SouceLength, H_num, H_dim -> B, H_num, SouceLength, H_dim
        query = self.query(q).view(batsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(k).view(batsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(v).view(batsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        x = self.attention_function(query, key, value)
        x = x.transpose(1, 2).contiguous().view(batsz, -1, self.d_model)
        x = self.out_linear(x)
        return x
    
class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model = 512, dim_feedforward = 2048):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layernorm = LayerNorm(d_model)
    
    def forward(self, x):
        src = self.sequential(x)
        return self.layernorm(x + src)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model = 512, num_heads = 8):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feedforward = PointWiseFeedForward(d_model)
        self.layernorm = LayerNorm(d_model)
        
    def forward(self, x):
        src = self.self_attn(x, x, x)
        src = self.layernorm(x + src)
        src = self.feedforward(src)
        return src
    
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer = TransformerEncoderLayer(), num_layers = 6):
        super().__init__()
        self.encoder_layer = nn.ModuleList([encoder_layer for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.encoder_layer:
            x = layer(x)
        return x

if __name__ == '__main__':
    # layernorm = LayerNorm()
    module = MultiHeadAttention(512, 8)
    # _layernorm = nn.LayerNorm(512)
    x = torch.rand([32, 10, 512])
    # print(_layernorm(x))
    print(module(x, x, x).shape)