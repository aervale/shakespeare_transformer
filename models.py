import os
import time

from config import config
from data_creation import *

layers = config['layers']
d_model = config['d_model']
heads = config['heads']
head_dim = config['head_dim']
block_size = config['block_size']
batch_size = config['batch_size']
dropout = config['dropout']
lr = config['lr']
ff_dim = config['ff_dim']

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__() #needed for parameter tracing
        self.k = nn.Linear(d_model, d_model, bias = False) #C x C
        self.q = nn.Linear(d_model, d_model, bias = False)
        self.v = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.full((block_size, block_size), float('-inf')), diagonal = 1))
        
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.k(x).view(B, T, heads, head_dim).transpose(1, 2)  #B x T x C -> B x T x heads x head_dim -> B x heads x T x head_dim
        q = self.q(x).view(B, T, heads, head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, heads, head_dim).transpose(1, 2)

        scale = C ** -0.5
        inner = (q @ k.transpose(-2, -1)) * scale #B x heads x T x head_dim @ B x heads x head_dim x T -> B x heads x T x T
        inner = inner + self.mask[:T, :T]        
        attn = torch.softmax(inner, dim=-1)
        attn = self.dropout(attn)
        attn = attn @ v                           #B x heads x T x T @ B x heads x T x head_dim -> B x heads x T x head_dim

        return attn.transpose(1, 2).contiguous().view(B, T, d_model)              #B x T x head_dim

class SingleHead(nn.Module):
    def __init__(self):
        super().__init__() #needed for parameter tracing
        self.k = nn.Linear(d_model, d_model, bias = False) #C x C
        self.q = nn.Linear(d_model, d_model, bias = False)
        self.v = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(d_model, d_model) #for the output of the multi head attention, we need to project it back to d_model

        self.register_buffer('mask', torch.triu(torch.full((block_size, block_size), float('-inf')), diagonal = 1))
        
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.k(x)                           #B x T x C
        q = self.q(x)
        v = self.v(x)

        scale = C ** -0.5
        inner = (q @ k.transpose(-2, -1)) * scale #B x T x T
        inner = inner + self.mask[:T, :T]         #B x T x T
        attn = torch.softmax(inner, dim=-1)
        attn = self.dropout(attn)

        return attn @ v                           #B x T x head_dim


class TransformerBlock(nn.Module): 
    def __init__(self):
        super().__init__()
        self.attn = MultiHead()
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        #x that is passed in is the embedding of the token and position
        x = x + self.attn(self.ln1(x)) #add the x+s for the residuals
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        #so first we need to embed, hen go through attention, add and nor (include residuals, then feed forward
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock() for i in range(layers)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device = idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.head(x)

        loss = None
        if (targets is not None):
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            context = idx[:, -block_size:]
            logits, loss = self(context)
            probs = F.softmax(logits[:, -1, :], dim =-1)
            sample = torch.multinomial(probs, num_samples = 1, replacement = True)
            idx = torch.cat([idx, sample], dim = 1)
        return idx
