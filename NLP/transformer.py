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
src_len = 5 # the length of input to encoder
src_voc_len = 10 # source vovabulary length
tgt_voc_len = 10 # target vocabulary length
tgt_len = 5 # the length of input to decoder
n_layer = 6

def PositionalEncoding_table(n_pos, d_model):
    def calc_angle(posi, i):
        if i % 2 != 0:
            i = i - 1
        return posi / np.power(10000, 2*i / d_model)
    
    def get_posi_vec(posi):
        return [calc_angle(posi, i) for i in range(d_model)] # a list with d_model length
    
    sinusoid_table = np.array([get_posi_vec(i) for i in range(n_pos)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # even index
    sinusoid_table[:, 1::2] = np.sin(sinusoid_table[:, 1::2]) # odd index
    return torch.FloatTensor(sinusoid_table)

def get_padding(q, k):
    # the shape of q: B, len_q
    # the shape of k: B, len_k
    batch_size, len_q = q.size()
    batch_size, len_k = k.size() 
    # if there are trivial words, set it true, later we will mask it
    padding_tensor = k.eq(0).unsqueeze(1) # B, 1, len_k
    padding_tensor = padding_tensor.expand(batch_size, len_q, len_k)
    return padding_tensor # B, len_q, len_k

def get_subsequent_padding(seq):
    seq_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(seq_shape), k = 1)
    return torch.from_numpy(subsequent_mask).byte() # uint8 type, upper limit is 255
    

class ScaledDot_Product(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q_s:torch.TensorType, k_s:torch.TensorType, v_s:torch.TensorType, mask):
        dot_product = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d_k)
        dot_product.masked_fill_(mask, 1e-9)
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
    def forward(self, q_input, k_input, v_input, mask):
        residue, batch_size = q_input, q_input.size(0)
        q_s = self.Q(q_input).view(batch_size, -1, heads, d_k).transpose(1, 2)
        k_s = self.K(k_input).view(batch_size, -1, heads, d_k).transpose(1, 2)
        v_s = self.V(v_input).view(batch_size, -1, heads, d_v).transpose(1, 2)
        mask = mask.unsqueeze(1).repeat(1, heads, 1, 1) # B, len_q, len_k -> B, H, len_q, len_k
        result = ScaledDot_Product()(q_s, k_s, v_s, mask)
        result = result.transpose(1, 2).reshape(batch_size, -1, heads * d_k)
        # print(result.is_contiguous())
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
    
    def forward(self, x, self_attn_mask):
        x = self.multihead(x, x, x, self_attn_mask)
        x = self.feedforward(x)
        return x
    
class Decoder_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_att = Multihead_Attention()
        self.dec_enc_att = Multihead_Attention()
        self.feedforward = PointWise_FeedForward()
    
    def forward(self, dec_input, enc_output, dec_self_mask, dec_enc_mask):
        dec_output = self.dec_self_att(dec_input, dec_input, dec_input, dec_self_mask)
        # print(dec_output.shape)
        dec_output = self.dec_enc_att(dec_output, enc_output, enc_output, dec_enc_mask)
        dec_output = self.feedforward(dec_output)
        return dec_output
    
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.voc_embedding = nn.Embedding(src_voc_len, d_model)
        self.positional_encode = nn.Embedding.from_pretrained(PositionalEncoding_table(src_len + 1, d_model), freeze=True)
        self.encoder_layer = nn.ModuleList([Encoder_Layer() for _ in  range(n_layer)])
    
    def forward(self, encode_input):
        sequence_number = [i for i in range(1, src_len)]
        sequence_number.append(0) # zero is for mask
        self_attn_mask = get_padding(encode_input, encode_input)
        encode_output = self.voc_embedding(encode_input) + self.positional_encode(torch.LongTensor(sequence_number))
        for enc_layer in self.encoder_layer:
            encode_output = enc_layer(encode_output, self_attn_mask)
        return encode_output
    
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.voc_embedding = nn.Embedding(tgt_voc_len, d_model)
        self.positional_encode = nn.Embedding.from_pretrained(PositionalEncoding_table(tgt_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([Decoder_Layer() for _ in range(n_layer)])
    
    def forward(self, dec_input, enc_output):
        sequence = [tgt_len]
        sequence.extend([i for i in range(1, tgt_len)])
        dec_outputs = self.voc_embedding(dec_input) + self.positional_encode(torch.LongTensor(sequence))
        self_dec_mask = get_subsequent_padding(dec_input)
        self_dec_pad_mask = get_padding(dec_input, dec_input)
        self_total_mask = torch.gt(self_dec_mask + self_dec_pad_mask, 0)
        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_output, self_total_mask, self_dec_pad_mask)
        return dec_outputs
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_voc_len)
    
    def forward(self, enc_input, dec_input):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_output)
        dec_output = self.projection(dec_output)
        return dec_output
        

if __name__ == '__main__':
    transformer = Transformer()
    tensor = torch.rand(1, 5).type(torch.LongTensor)
    print(transformer(tensor, tensor).shape)
    
    # print(PositionalEncoding_table(src_len, d_model).shape)