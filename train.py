import os
import time
import math
import pickle
from contextlib import nullcontext
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tokenizer import *
from eval import *
from pencil_utils import *

# -----------------------------------------------------------------------------
DEBUG = False
format = 'pencil' # 'cot' or 'pencil'
# training settings
eval_unit = 'iters'  # or 'flops'
eval_interval = 1000  
max_passes = 1  # number of passes through the whole training data
max_nsubseq = 500
# -----------------------------------------------------------------------------
# evaluation settings
eval = True # evaluate the model
eval_while_train = False # evaluate while training, for debugging purpose, will slow down training
eval_trace = False # print out the trace rate
max_new_tokens = 50000 # forced stop generating after this many tokens
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
num_samples = 100  # number of samples to evaluate
# -----------------------------------------------------------------------------
# logging
wandb_log = False # disabled by default
wandb_project = 'PENCIL'
wandb_run_name = 'run' + str(time.time()) # 'run' + str(time.time())
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
log_interval = 1
eval_only = False # if True, script exits right after the first eval
# always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = '3sat'
data_dir = 'data'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# out_dir = os.path.join('out', dataset)
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# print what device we're using
print("running on device:", device)

# Initialize wandb logging if enabled
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, group=dataset, name=wandb_run_name, 
               config=config, job_type='training')

# Load data and tokenizer
class DataLoader:
    def __init__(self, data_dir, format='io'):
        self.data_dir = data_dir
        self.format = format
        self.current_file_idx = 0
        self.train_files = sorted([
            f for f in os.listdir(data_dir) 
            if f.startswith('pencil' if format == 'pencil' else 'train') 
            and f.endswith('.bin')
        ])
        if not self.train_files:
            raise ValueError(f"No training files found in {data_dir}")
        self.load_next_file()
        
    def load_next_file(self):
        if self.current_file_idx >= len(self.train_files):
            print("All training files processed. Training complete.")
            return False
            
        file_path = os.path.join(self.data_dir, self.train_files[self.current_file_idx])
        print(f"Loading training file {self.train_files[self.current_file_idx]}...")
        with open(file_path, 'rb') as f:
            self.current_data = pickle.load(f)
        self.current_file_idx += 1
        return True

    def get_current_data(self):
        return self.current_data

# Modified training data loading section
data_dir = os.path.join(data_dir, dataset)

# Initialize data loader
train_loader = DataLoader(data_dir, format)
train_data = train_loader.get_current_data()

# Load validation data and tokenizer
with open(os.path.join(data_dir, 'val.bin'), 'rb') as f:
    val_data = pickle.load(f)
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)    
tokenizer = SimpleTokenizer().load_from_meta(meta)

print(f"Loaded first training batch with {len(train_data)} sequences")

# ---------------------------------------------
# 1) Multi-pass logic is placed in get_batch
# ---------------------------------------------
def get_batch(split):
    global train_data, train_loader

    # We choose how many times we want to pass through each file
    MAX_PASSES = max_passes

    # Static variables for controlling the data flow
    if not hasattr(get_batch, '_state'):
        get_batch._state = {
            'idx': 0,         # current index in train_data
            'pass_count': 0,  # how many passes on current file
            'max_passes': MAX_PASSES
        }
    state = get_batch._state

    #  Step A: Check if we’ve exhausted current file
    if state['idx'] >= len(train_data):
        # We finished one full pass through the file
        state['pass_count'] += 1

        if state['pass_count'] >= state['max_passes']:
            # Try loading the next file
            loaded = train_loader.load_next_file()
            if not loaded:
                return None, None   # No more data => end training

            # Reset counters for the new file
            train_data = train_loader.get_current_data()
            state['pass_count'] = 0
            state['idx'] = 0
        else:
            # Haven’t hit max passes yet => just reset idx
            state['idx'] = 0

    #  Step B: Collect batch indices
    current_idx = state['idx']
    end_idx = min(current_idx + batch_size, len(train_data))
    batch_indices = list(range(current_idx, end_idx))

    # Update state to point to next batch
    state['idx'] = end_idx

    if len(batch_indices) == 0:
        # In a rare corner case: if the file is empty or something
        return None, None

    #  Step C: Call the format-specific function to build x, y
    if format == 'pencil':
        x, y = get_batch_pencil(batch_indices)
    else:
        x, y = get_batch_sequence(batch_indices)

    return x, y

# -----------------------------------------------------------------------------
# Batch builders
# -----------------------------------------------------------------------------
def get_batch_sequence(batch_indices):
    sequences = []
    for idx in batch_indices:
        seq = train_data[idx]
        if len(seq) > block_size + 1:
            # ...
            seq = seq[:block_size + 1]
        sequences.append(seq)

    # create x, y from sequences
    max_len = max(len(seq) for seq in sequences)
    x = torch.zeros((len(sequences), max_len - 1), dtype=torch.long)
    y = torch.zeros_like(x)

    prompt_token_id = meta['stoi']['<|endofprompt|>']
    for i, seq in enumerate(sequences):
        x[i, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
        y[i, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)
        prompt_pos = (x[i] == prompt_token_id).nonzero()
        if len(prompt_pos) > 0:
            y[i, :prompt_pos[0]] = 0

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

# ---------------------------------------------
# 3) Format-specific: get_batch_pencil
# ---------------------------------------------
def get_batch_pencil(batch_indices):
    subinstances = [s for seq in (train_data[i] for i in batch_indices) for s in seq]
    random.shuffle(subinstances)
    subinstances = subinstances[:max_nsubseq]
    
    sequences = [d['ids'][:block_size + 1] for d in subinstances]
    mask_indices = [d['mask_idx'] for d in subinstances]

    max_len = max(len(seq) for seq in sequences)
    x = torch.zeros((len(sequences), max_len - 1), dtype=torch.long)
    y = torch.zeros_like(x)

    for i, seq in enumerate(sequences):
        x[i, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
        y[i, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)
        y[i, :mask_indices[i]] = 0

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # from command line or config

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta['vocab_size']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
else:
    raise ValueError(f"unrecognized init_from {init_from}")
assert block_size == model.config.block_size
model.to(device)

# optimizer & GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0+

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
# prime the first batch
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

iter_num = 0
total_flops = 0
best_val_loss = 1e9
training_losses = []
last_eval_step = 0      # tracks last multiple where we did an eval

while True:
    # If we only want a single eval (eval_only) right at start, then exit
    if iter_num == 0 and eval_only:
        break

    # 1) Set learning rate for this iteration
    if decay_lr:
        lr = get_lr(iter_num)
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2) Training step
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            training_losses.append(loss.item())

            # scale down loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # compute flops for this batch
            flops = calculate_flops(
                X, Y,
                tokenizer.word2idx[PENCIL_TOKENS['sep']],
                tokenizer.word2idx[PENCIL_TOKENS['call']],
                tokenizer.word2idx[PENCIL_TOKENS['return']],
                format=format
            )
            total_flops += flops
        # fetch the next micro-batch
        X, Y = get_batch('train')
        if X is None:
            # data exhausted mid micro-step
            break
        
        # backward pass
        scaler.scale(loss).backward()

    if X is None:
        # entire data is exhausted
        break

    # 3) Optimizer step
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 4) Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        # un-averaged loss from the last micro-step
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.5f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # 5) Evaluation checks
    #    - Evaluate at intervals of eval_interval in the chosen eval_unit
    current_progress = total_flops if eval_unit == 'flops' else iter_num
    # If we've reached or passed the next multiple of eval_interval, do an eval
    if current_progress >= last_eval_step + eval_interval:
        # ensure we do the eval once per multiple, even if we skip across intervals
        while current_progress >= last_eval_step + eval_interval:
            last_eval_step += eval_interval
        print(f"\nEvaluation at {current_progress} {eval_unit}...")
        
        avg_loss = sum(training_losses[-eval_interval:]) / len(training_losses[-eval_interval:])

        # do the evaluation
        args = dict(
            model=model,
            val_data=val_data,
            tokenizer=tokenizer,
            ctx=ctx,
            format=format,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            num_samples=num_samples,
            out_dir=out_dir,
            trace_rate=eval_trace,
            progress=current_progress,
            training_loss=avg_loss,
            iter=iter_num
        )
        results = evaluate_model(**args)

        if eval_trace:
            accuracy, trace_rate = results
        else:
            accuracy = results
            trace_rate = None

        if wandb_log and master_process:
            log_dict = {
                "eval_progress": current_progress,
                "accuracy": accuracy
            }
            if trace_rate is not None:
                log_dict["trace_rate"] = trace_rate
            wandb.log(log_dict)

# -----------------------------------------------------------------------------
# After the loop ends, all data is exhausted
# -----------------------------------------------------------------------------
print("\nAll data exhausted. Training complete.")

# optional final evaluation
if eval:
    print("Final Evaluation")
    args = dict(
        model=model, 
        val_data=val_data, 
        tokenizer=tokenizer, 
        ctx=ctx, 
        format=format,
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k,
        num_samples=num_samples, 
        out_dir=out_dir, 
        trace_rate=eval_trace, 
        progress=last_eval_step,
        iter=iter_num
    )
    results = evaluate_model(**args)
    if eval_trace:
        accuracy, trace_rate = results
        print(f"Final accuracy = {accuracy}, final trace rate = {trace_rate}")
    else:
        accuracy = results
        print(f"Final accuracy = {accuracy}")

    if wandb_log and master_process:
        wandb.log({"final_accuracy": accuracy})
        if eval_trace:
            wandb.log({"final_trace_rate": trace_rate})
        wandb.run.summary["final_accuracy"] = accuracy
        if eval_trace:
            wandb.run.summary["final_trace_rate"] = trace_rate
        wandb.finish()

# cleanup if using ddp
if ddp:
    destroy_process_group()