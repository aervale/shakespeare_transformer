import torch
import torch.nn as nn
import torch.nn.functional as F

import model

if __name__ == "__main__":
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
  
    # generation
    prompt = torch.zeros((1, 1), dtype=torch.long).to(torch.device)
    out = model.generate(prompt, max_new_tokens=2000)
    print(model.decode(list(i.item() for i in out[0])))    # print(f"Generated tokens: {decode(list(out))}")