config = {
  "layers": 2, #how many times to run TransformerBlock
  "d_model": 1024, #the same as nembd, C
  "heads": 2,
  "head_dim": 512, #gonna be the same as d_model for the case with 1 head
  "block_size": 64, #T
  "batch_size": 16, #B
  "dropout": 0.1, #percentage dropped
  "lr": 0.01, #learning rate
  "ff_dim": 256, #d_model * 4
  "maxiters": 100000,
  "eval_interval": 100
}

config_path = 'config.py'