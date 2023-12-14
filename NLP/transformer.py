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
        return self.beta * (x - x_mean) / (x_std + self.eps) + self.gamma

def PositionalEncoder(n_posi, d_model):
    # compute one angle of the item
    def calc_angle(posi, i):
        if i % 2 != 0:
            i = i - 1
        return posi / np.power(10e4, 2 * i / d_model)
    
    # computer one position of encode
    def calc_positional_mask(posi):
        return [calc_angle(posi, i) for i in range(d_model) ]
    
    
    positional_mask = np.array([calc_positional_mask(i) for i in range(n_posi)])
    positional_mask[:, 0::2] = np.sin(positional_mask[:, 0::2])
    positional_mask[:, 1::2] = np.cos(positional_mask[:, 1::2])
    return torch.from_numpy(positional_mask)

def ShiftRightMask(seq):
    """generate mask

    Args:
        seq (tensor): B x Seq_len x d_model

    Returns:
        mask: tensor, B x Seq_len x Seq_len
    """
    
    b, n = seq.size(0), seq.size(1)
    masked = np.ones([b, n, n])
    masked = np.triu(masked, k = 1)
    return torch.from_numpy(masked)


class ScaledDotProduct(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, q, k, v, mask):
        d_k = q.size(-1)
        # B, H_num, L, H_dim
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)
        
        if mask is not None:
            scores.masked_fill_(mask, -1e9)
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
        
    def forward(self, q, k, v, mask=None):
        batsz = q.size(0)
        # B, SouceLength, d_model -> B, SouceLength, H_num, H_dim -> B, H_num, SouceLength, H_dim
        query = self.query(q).view(batsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(k).view(batsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(v).view(batsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        "(B, SouceLength, SouceLength) -> (B, 1, SouceLength, SouceLength) -> B, H_num, SouceLength, SouceLength"
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        x = self.attention_function(query, key, value, mask) 
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
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self,d_model = 512, n_head = 8, dim_feedforward = 2048, masked = None):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_head)
        self.layernorm2 = LayerNorm(d_model)
        self.ff = PointWiseFeedForward(d_model, dim_feedforward)
        
    def forward(self, enc_input, dec_input, mask=None):
        src = dec_input
        dec_output = self.dec_self_attn(dec_input, dec_input, dec_input, mask)
        dec_output = self.layernorm1(src + dec_input)
        src = dec_output
        dec_output = self.enc_dec_attn(dec_output, enc_input, enc_input)
        dec_output = self.layernorm2(src + dec_output)
        dec_output = self.ff(dec_output)
        return dec_output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer = TransformerDecoderLayer(), num_layers = 6, mask=None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.mask = mask
        
    def forward(self, enc_output, dec_input):
        if self.mask is None:
            dec_self_mask = ShiftRightMask(dec_input)
            dec_self_mask = dec_self_mask.gt(0)
        else:
            dec_self_mask = self.mask
            
        for layer in self.layers:
            dec_output = layer(enc_output, dec_input, dec_self_mask)
            dec_input = dec_output
        return dec_input
        
class Transformer(nn.Module):
    def __init__(self,d_model = 512, nhead = 8, num_encoder_layers = 6, num_decoder_layers = 6):
        super().__init__()
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nhead), num_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model, nhead), num_decoder_layers)
        
    def forward(self, enc_input, dec_input):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(enc_output, dec_input)
        return dec_output


if __name__ == '__main__':
    
    module = Transformer()
    x = torch.rand([32, 10, 512])
    print(module(x, x).shape)