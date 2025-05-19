import torch
import torch.nn as nn
import torch.nn.functional as F
import math


VOCAB_SIZE = 50257 
MAX_SEQ_LEN = 1024 
D_MODEL = 768   
NUM_HEADS = 12   
NUM_LAYERS = 12  
DROPOUT_RATE = 0.1 

class SingleAttention(nn.Module):
  
  def __init__(self):
    super().__init__()

  def forward(self, q, k, v, dropout=None, masked=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    seq_len = q.size(-2)

    if masked:
      causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
      scores = scores.masked_fill(~causal_mask, float('-inf'))

    scores = F.softmax(scores, dim=-1)

    if dropout:
      scores = dropout(scores)

    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):

  def __init__(self, num_head, d_model, dropout):
    super().__init__()
    self.num_head = num_head
    self.d_model = d_model
    assert d_model % num_head == 0
    self.head_dim = d_model // num_head

    self.q_lin = nn.Linear(d_model, d_model)
    self.k_lin = nn.Linear(d_model, d_model)
    self.v_lin = nn.Linear(d_model, d_model)
    self.out_lin = nn.Linear(d_model, d_model)

    self.attention = SingleAttention()
    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, mask=False):
    batch_size = q.size(0)

    q_proj = self.q_lin(q).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
    k_proj = self.k_lin(k).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
    v_proj = self.v_lin(v).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)

    attn = self.attention(q_proj, k_proj, v_proj, dropout=self.dropout, masked=mask)

    out = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    return self.out_lin(out)




class LayerNorm(nn.Module):

  def __init__(self, d_model, eps=1e-5):
      super().__init__()
      self.gamma = nn.Parameter(torch.ones(d_model))
      self.beta = nn.Parameter(torch.zeros(d_model))
      self.eps = eps

  def forward(self, x):
      mean = x.mean(-1, keepdim=True)
      std = x.std(-1, keepdim=True, unbiased=False)
      return self.gamma * (x - mean) / (std + self.eps) + self.beta


class GELU(nn.Module):

  def forward(self,x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(2/3.142) * (x+0.044715*x**3)))


class FeedForward(nn.Module):

  def __init__(self,d_model,d_ff,dropout=0.1):
    super().__init__()
    self.lin_1 = nn.Linear(d_model,d_ff)
    self.lin_2 = nn.Linear(d_ff,d_model)
    self.gelu = GELU()
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    return self.lin_2(self.dropout(self.gelu(self.lin_1(x))))



class PosEmb(nn.Module):

  def __init__(self,d_model,max_len=512):
    super().__init__()
    self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

  def forward(self, x):
    seq_len = x.size(1)
    pos_emb = self.pe[:seq_len, :].unsqueeze(0)
    return x + pos_emb



class Residual(nn.Module):

  def __init__(self, size, dropout):
      super().__init__()
      self.norm = LayerNorm(size)
      self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
      return x + self.dropout(sublayer(self.norm(x)))





class DecoderLayer(nn.Module):

  def __init__(self, d_model, d_ff, num_head, dropout):
    super().__init__()
    self.mha = MultiHeadAttention(num_head,d_model,dropout)
    self.ff = FeedForward(d_model,d_ff)
    self.residual1 = Residual(d_model,dropout)
    self.residual2 = Residual(d_model,dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):

    x = self.residual1(x,lambda x: self.mha(x,x,x,mask=True))
    x = self.residual2(x,self.ff)
    return x


class GPT2Model(nn.Module):

  def __init__(self, vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers, dropout,eps):
      super().__init__()
      self.token_emb = nn.Embedding(vocab_size, d_model)
      self.pos_emb = PosEmb(d_model, max_seq_len)
      self.layers = nn.ModuleList(
          [DecoderLayer(d_model, d_ff,num_heads, dropout) for _ in range(num_layers)]
      )
      self.ln_f = LayerNorm(d_model,eps=eps)  

  def forward(self, input_ids):
      x = self.token_emb(input_ids)           
      x = self.pos_emb(x)                      
      for layer in self.layers:
          x = layer(x)
      x = self.ln_f(x)
      return x
