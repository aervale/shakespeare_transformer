
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os

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
