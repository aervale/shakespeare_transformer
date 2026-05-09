import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

shakespeare_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
save_path = os.path.join(os.path.dirname(__file__), 'input.txt')
urllib.request.urlretrieve(shakespeare_url, save_path)
docs = open(save_path, 'r').read()
docs = [line.strip() for line in open(save_path) if line.strip()]

data = '\n'.join(docs)

chars = sorted(list(set(data))) #sorted list of all the letters used
vocab_size = len(chars) 

#embedding of the characters
stoi = {s:i for i, s in enumerate(chars)} #convert strings to integers
itos = {i:s for s, i in stoi.items()} #reverse stoi
encode = lambda s: [stoi[ch] for ch in s] #for a string convert to list of ints
decode = lambda i: ''.join([itos[num] for num in i]) #for a list of integers convert them to numbers

data = encode('\n'.join(docs))

#divide the data set into training and dev sets
n = int(len(data) * 0.9)
trdata = torch.tensor(data[:n])
devdata = torch.tensor(data[n:])

layers = 4 #how many times to run TransformerBlock

d_model = 64 #the same as nembd, C
head_dim =  d_model #gonna be the same as d_model for the case with 1 head
block_size = 128 #T
batch_size = 32

dropout = 0.1 #percentage dropped
lr = 0.01 #learning rate
ff_dim = d_model * 4
maxiters = 100000

class SingleHead(nn.Module):
    def __init__(self):
        super().__init__() #needed for parameter tracing
        self.k = nn.Linear(d_model, d_model, bias = False) #C x C
        self.q = nn.Linear(d_model, d_model, bias = False)
        self.v = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.full((block_size, block_size), float('-inf')), diagonal = 1))
        
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.k(x)                             #B x T x C
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
        self.attn = SingleHead()
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
    
model = Transformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
lossi = []
for i in range (maxiters):
    ix = torch.randint(0, len(trdata) - block_size, (batch_size,)) #shape [32]
    x = torch.stack([trdata[i : i + block_size] for i in ix]).to(device)      #shape [32, 128], batch_size x block_size
    y = torch.stack([trdata[i+1 : i + 1 + block_size] for i in ix]).to(device)
    
    logits, loss = model(x, y)
    lossi.append(loss.item())

    if i % 100 == 0:
        print(f'trial {i}/{maxiters}: loss = {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
def get_batch(data):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1 : i + 1 + block_size] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, eval_iters=200):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

print(f'{estimate_loss(model, trdata):.4f}, {estimate_loss(model, devdata):.4f}')

if __name__ == "__main__":
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
  
    # generation
    prompt = torch.zeros((1, 1), dtype=torch.long).to(device)
    out = model.generate(prompt, max_new_tokens=2000)
    print(decode(list(i.item() for i in out[0])))    # print(f"Generated tokens: {decode(list(out))}")