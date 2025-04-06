eval_unit = 'iters'
eval_interval = 1000 # keep frequent because we'll overfit
log_interval = 10 # don't print too too often

dataset = '3sat'
gradient_accumulation_steps = 4 # set it larger to simulate larger batch size
batch_size = 5
block_size = 2048 # context of up to 2048 previous tokens

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'mps'  # run on cpu only
compile = False # do not torch compile the model