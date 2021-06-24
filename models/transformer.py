"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self,block_size, n_head=8, n_embed=256, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Attention, self).__init__()
        assert n_embed % n_head == 0
        self.q = nn.Linear(n_embed, n_embed)
        self.k = nn.Linear(n_embed, n_embed)
        self.v = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embed, n_embed)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()
        q_ = self.q(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2) # (b, nh, t, hs)
        k_ = self.k(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v_ = self.v(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        # (b, nh, t, hs) x (b, nh, ns, t) -> (b, nh, t, t)
        att = (q_ @ k_.transpose(-2, -1)) * (1. / math.sqrt(k_.size(-1)))
        att = att.masked_fill(self.mask[:, :, t, t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v_ # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, block_size,n_head=8, n_embed=256, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = Attention(block_size,n_head, n_embed, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(resid_pdrop)
        )

    def forward(self, x):
        x += self.attn(self.ln1(x))
        x += self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    vocab_size: 1024 (equals to codebook size)
    block_size: 512
    """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embed=256,
                 embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_unmasked=0):
        super(GPT, self).__init__()
        # input embedding
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embed))
        self.drop = nn.Dropout(embed_pdrop)
        # transformer
        self.layers = nn.Sequential(*[Block(block_size) for _ in range(n_layer)])
        # decoder head
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)
        