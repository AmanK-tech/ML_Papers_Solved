import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleAttention(nn.Module):
  def forward(self, q, k, v, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, v)



class MultiHeadAttention(nn.Module):
  def __init__(self,num_head,d_model,dropout):
    super().__init__()
    self.num_head = num_head
    self.d_model = d_model
    self.h = d_model // num_head
    self.q_lin = nn.Linear(self.d_model,self.d_model)
    self.k_lin  = nn.Linear(self.d_model,self.d_model)
    self.v_lin  = nn.Linear(self.d_model,self.d_model)
    self.out_lin  = nn.Linear(self.d_model,self.d_model)
    self.attention = SingleAttention()
    self.dropout = nn.Dropout(dropout)


  def forward (self,q,k,v,mask=False):
    q_proj = self.q_lin(q)
    k_proj = self.k_lin(k)
    v_proj = self.v_lin(v)

    q_proj = q_proj.view(q_proj.size(0),q_proj.size(1),self.num_head,self.h)
    q_proj = q_proj.transpose(1,2)
    k_proj = k_proj.view(k_proj.size(0),k_proj.size(1),self.num_head,self.h)
    k_proj = k_proj.transpose(1,2)
    v_proj = v_proj.view(v_proj.size(0),v_proj.size(1),self.num_head,self.h)
    v_proj = v_proj.transpose(1,2)


    attn = self.attention(q_proj,k_proj,v_proj,mask,self.dropout)

    out = attn.transpose(1,2)


    out = out.contiguous().view(out.size(0),out.size(1),-1)

    return self.out_lin(out)



class LayerNorm(nn.Module):
  def __init__(self,d_model,eps=1e-6):
    super().__init__()
    self.d_model = d_model
    self.gamma = nn.Parameter(torch.ones(d_model))
    self.beta = nn.Parameter(torch.zeros(d_model))
    self.eps = eps

  def forward(self,x):
    mean = x.mean(-1,keepdim=True)
    std = x.std(-1,keepdim=True,unbiased=False)

    return self.gamma * (x - mean) / (std + self.eps) + self.beta



class Residual(nn.Module):

  def __init__(self, size, dropout):
      super().__init__()
      self.norm = LayerNorm(size)
      self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
      return x + self.dropout(sublayer(self.norm(x)))




class GELU(nn.Module):
  def forward(self,x):
    return 0.5 * x *(1+torch.tanh(torch.sqrt(22/7)*(x+0.044715 * x**3)))



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
    return self.pe[:seq_len, :]




class TokenEmb(nn.Module):

  def __init__(self,vocab_size,d_model):
    super().__init__()
    self.emb = nn.Embedding(vocab_size,d_model)

  def forward(self,input_ids):
    return self.emb(input_ids)





class SegEmb(nn.Module):

  def __init__(self,vocab_size,d_model):
    super().__init__()
    self.emb = nn.Embedding(vocab_size,d_model)

  def forward(self,token_type_ids):
    return self.emb(token_type_ids)






class BertEmb(nn.Module):

  def __init__(self,vocab_size,d_model=512,max_seq_len=512):
    super().__init__()

    self.SegEmbd = SegEmb(vocab_size,d_model)
    self.TokenEmbd = TokenEmb(vocab_size,d_model)
    self.PosEmbd = PosEmb(d_model,max_seq_len)
    self.dropout = nn.Dropout(0.1)

  def forward(self, input_ids, token_type_ids):
      token_emb = self.TokenEmbd(input_ids)
      segment_emb = self.SegEmbd(token_type_ids)
      position_emb = self.PosEmbd(token_emb)

      embeddings = token_emb + segment_emb + position_emb

      return self.dropout(embeddings)


class EncoderLayer(nn.Module):
  def __init__(self,hidden,ff_hidden,num_head,dropout):
    super().__init__()

    self.attention = MultiHeadAttention(num_head,hidden,dropout)
    self.feed_forward = FeedForward(hidden,ff_hidden)
    self.residual1 = Residual(hidden,dropout=dropout)
    self.residual2 = Residual(hidden,dropout=dropout)

    self.dropout = nn.Dropout(dropout)

  def forward(self,x,mask):
    x = self.residual1(x,lambda x: self.attention(x,x,x,mask))
    x = self.residual2(x, self.feed_forward)
    return x


class BERT(nn.Module):
  def __init__(self,vocab_size,hidden=768,num_layers=12, num_head = 12, dropout=0.1,):
    super().__init__()
    self.hidden = hidden
    self.num_layers = num_layers
    self.num_head = num_head
    self.dropout = nn.Dropout(dropout)

    self.embeddings = BertEmb(vocab_size=vocab_size,d_model=hidden)
    self.layers = nn.ModuleList(EncoderLayer(hidden=hidden,ff_hidden=hidden*4,num_head=num_head,dropout=dropout)for _ in range(num_layers))

  def forward(self, x, seg):
    mask = (x != 0).unsqueeze(1).unsqueeze(2)
    x = self.embeddings(x, seg)
    for layer in self.layers:
        x = layer(x, mask)
    return x




