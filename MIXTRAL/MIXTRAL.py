import torch
import torch.nn as nn
import torch.nn.functional as F
import math


DIM = 4096
n_layers = 32
head_dim = 128
hidden_dim = 14336
n_heads = 32
n_kv_heads = 8
context_len = 32768
vocab_size = 32000
num_experts = 8
top_k_experts = 2


class SingleAttention(nn.Module):

  def forward(self, q, k, v, dropout=None, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
    seq_len = q.size(-2)

    if mask:
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))

    scores = F.softmax(scores, -1)

    if dropout:
        scores = dropout(scores)

    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):

  def __init__(self, n_heads, d_model, dropout):
    super().__init__()
    self.n_heads = n_heads
    self.d_model = d_model
    assert d_model % n_heads == 0
    self.head_dim = d_model // n_heads

    self.q_lin = nn.Linear(d_model, d_model)
    self.k_lin = nn.Linear(d_model, d_model)
    self.v_lin = nn.Linear(d_model, d_model)
    self.out_lin = nn.Linear(d_model, d_model)

    self.attn = SingleAttention()
    self.dropout = nn.Dropout(dropout)
    self.rope = RoPE(self.head_dim)

  def forward(self, q, k, v, mask=None):

    batch_size = q.size(0)

    q_proj = self.q_lin(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    k_proj = self.k_lin(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    v_proj = self.v_lin(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

    q_proj, k_proj = self.rope(q_proj, k_proj)

    mha = self.attn(q_proj, k_proj, v_proj, dropout=self.dropout, mask=mask)

    out = mha.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    return self.out_lin(out)


class GatingFunction(nn.Module):


  def __init__(self, top_k, num_experts, d_model):
    super().__init__()
    self.num_experts = num_experts
    self.top_k = top_k
    self.lin = nn.Linear(d_model, num_experts)

  def forward(self, x):


    logits = self.lin(x)

    values, indices = torch.topk(logits, self.top_k, dim=-1)

    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, indices, values)

    weights = F.softmax(mask, dim=-1)

    return weights, indices


class SwiGLU(nn.Module):


  def __init__(self, dim, hidden_dim):
      super().__init__()
      self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
      self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
      self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x):
      gate = self.gate_proj(x)
      up = self.up_proj(x)
      swish_gate = gate * torch.sigmoid(gate)
      return self.down_proj(swish_gate * up)


class MoeBlock(nn.Module):
  def __init__(self, dim, hidden_dim, top_k, num_experts):
      super().__init__()
      self.gating = GatingFunction(top_k, num_experts, dim)
      self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])
      self.top_k = top_k
      self.num_experts = num_experts

  def forward(self, x):
    batch_size, seq_len, dim = x.shape
    x_flat = x.view(-1, dim)

    weights, indices = self.gating(x_flat)

    output = torch.zeros_like(x_flat)

    for expert_id in range(self.num_experts):
        expert_mask = (indices == expert_id).any(dim=-1)

        if expert_mask.any():
            expert_input = x_flat[expert_mask]
            expert_output = self.experts[expert_id](expert_input)

            expert_weights = weights[expert_mask, expert_id]

            output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)

    return output.view(batch_size, seq_len, dim)

class RoPE(nn.Module):

  def __init__(self, head_dim, max_seq_len=32768, base=10000):

    super().__init__()
    self.head_dim = head_dim
    self.max_seq_len = max_seq_len
    self.base = base
    assert head_dim %2 == 0

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    self.register_buffer('inv_freq',inv_freq,persistent=False)
    self._precompute_cos_sin(max_seq_len)

  def _precompute_cos_sin(self, seq_len):

    position = torch.arange(seq_len).float()

    angles = torch.outer(position,self.inv_freq)

    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    self.register_buffer('cos_cached', cos_vals, persistent=False)
    self.register_buffer('sin_cached', sin_vals, persistent=False)


  def forward(self, q, k, start_pos=0):
    seq_len = q.size(1)

    if start_pos + seq_len > self.cos_cached.size(0):
      self._precompute_cos_sin(start_pos + seq_len)


    cos = self.cos_cached[start_pos:start_pos + seq_len]
    sin = self.sin_cached[start_pos:start_pos + seq_len]

    q_rotated = self._apply_rotation(q, cos, sin)
    k_rotated = self._apply_rotation(k, cos, sin)

    return q_rotated, k_rotated

  def _apply_rotation(self,x,cos,sin):

    x1 = x[...,::2]
    x2 = x[..., 1::2]

    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)


    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x1 * sin + x2 * cos

    rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)  
    rotated = rotated.flatten(start_dim=-2) 

    return rotated


class RMSNorm(nn.Module):
  def __init__(self,d_model,epsilon=1e-8):
    super().__init__()
    self.epsilon = epsilon
    self.d_model = d_model
    self.weight = nn.parameter(torch.ones(d_model))

  def forward(self,x):
    rms = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.epsilon)
    yi = x/rms

    return yi * self.weight


class Residual(nn.Module):
  def __init__(self,size,dropout):
    super().__init__()
    self.norm = RMSNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x,sublayer):
    return x + self.dropout(sublayer(self.norm(x)))



class Decoder(nn.Module):
  def __init__(self,num_head,d_model,dropout,hidden_dim,top_k):
    super().__init__()
    self.mha = MultiHeadAttention(num_head,d_model,dropout)
    self.moe = MoeBlock(d_model,hidden_dim,top_k,num_experts)
    self.residual1 = Residual(d_model,dropout)
    self.residual2 = Residual(d_model,dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):

    x = self.residual1(x, lambda x: self.mha(x,x,x,mask=True))
    x = self.residual2(x,self.moe)

    return x

class MOEModel(nn.Module):

  def __init__(self, d_model, max_seq_len, n_layers):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, d_model)
    self.layers = nn.ModuleList([
        Decoder(n_heads, d_model, dropout=0.1, hidden_dim=hidden_dim, top_k=top_k_experts)
        for _ in range(n_layers)
    ])
    self.norm = RMSNorm(d_model)
    self.output_layer = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    x = self.token_embedding(x)
    for layer in self.layers:
        x = layer(x)
    x = self.norm(x)
    logits = self.output_layer(x)
    return logits






