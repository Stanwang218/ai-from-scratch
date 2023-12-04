import torch
import torch.nn as nn

#data flow: 

# after embedding: batch_size, source_len, d_model

d_model = 512 # length of embedding
d_k = 8 # length of Q and K
d_v = 8 # length of V
heads = 6 # number of heads
batch_size = 100

class ScaledDot_Production(nn.Module):
    def __init__(self):
        super.__init__()
    
    def forward(self, q_s, k_s, v_s):
        dot_product = torch.matmul(q_s, k_s.transpose(-1, -2)) / torch.sqrt(d_k)
        # mask 
        dot_product = nn.Softmax(dim=-1)
        result = torch.matmul(dot_product, v_s)
        # (B, H, S, S)
        return result

class Multihead_Attention(nn.Module):
    def __init__(self) -> None:
        # input: batch_size(B), source_len(S), d_model(D)
        # Q, k, V
        super().__init__()
        self.Q = nn.Linear(d_model, d_k * heads) # batch, source
        self.K = nn.Linear(d_model, d_k * heads)
        self.V = nn.Linear(d_model, d_v * heads)
        self.linear = nn.Linear(heads * d_k, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    # (B, S, D(model)) -project--> (B, S, D(k) * H) -split--> (B, S, H, D(K)) -transpose--> (B, H, S, D(K))
    def forward(self, q_input, k_input, v_input):
        residue, batch_size = q_input, q_input.size(0)
        q_s = self.Q(q_input).view(batch_size, -1, heads, d_k).transpose(1, 2)
        k_s = self.K(k_input).view(batch_size, -1, heads, d_k).transpose(1, 2)
        v_s = self.V(v_input).view(batch_size, -1, heads, d_v).transpose(1, 2)
        result = ScaledDot_Production()(q_s, k_s, v_s)
        result = result.transpose(1, 2).view(batch_size, -1, heads * d_k)
        result = self.linear(result)
        return self.layer_norm(residue + result)
        
if __name__ == '__main__':
    t = torch.rand((2, 512, 10))
    l = nn.Linear(10, 20)
    print(l(t).shape)