eval_unit = 'iters'
eval_interval = 5000 # keep frequent because we'll overfit
log_interval = 10 # don't print too too often

dataset = 'puzzle'
gradient_accumulation_steps = 10 # set it larger to simulate larger batch size
batch_size = 1
block_size = 2048 # context of up to 2048 previous tokens
max_new_tokens=350000 # maximum number of tokens to generate in each sample

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
lr_decay_iters = 20000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially

max_passes = 2
max_nsubseq = 50

# on macbook also add
device = 'mps'  # run on cpu only
compile = False # do not torch compile the model