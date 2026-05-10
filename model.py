import matplotlib.pyplot as plt
import torch
import time

from config import *
from data_creation import *
from models import *
from log import write

best_loss = float('inf')
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

start_time = time.time()

model = Transformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

eval_interval = config['eval_interval']
maxiters = config['maxiters']

def get_batch(data):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1 : i + 1 + block_size] for i in ix]).to(device)
    return x, y

lossi = []
for i in range (maxiters):
    x, y = get_batch(trdata)
    
    logits, loss = model(x, y)
    lossi.append(loss.item())

    if (i % eval_interval == 0):
        x, y = get_batch(devdata)
        _, devloss = model(x, y)
        print(f"step {i}: dev loss {devloss:.4f}")
        if (devloss < best_loss):
            best_loss = devloss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': devloss,
                'config': config,
                "step": i
            }
            torch.save(checkpoint, 'checkpoint.pth')
            print(f"New best model saved at step {i} with dev loss {devloss:.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end_time = time.time()

@torch.no_grad()
def estimate_loss(model, data, eval_iters=1000):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

print("------------------stats------------------")
print(f"Final loss: {loss.item():.4f}")
print(f"Training time: {end_time - start_time:.2f} seconds")
trloss = estimate_loss(model, trdata)
devloss = estimate_loss(model , devdata)
print(f'{trloss}, {devloss}')
print(f'{config}')
print("-----------------------------------------")
write(end_time - start_time, devloss, sum(p.numel() for p in model.parameters()), config, devloss-trloss)