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
max_passes = 1
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

# -----------------------------------------------------------------------------
# Round-robin DataLoader
# -----------------------------------------------------------------------------
class RoundRobinDataLoader:
    """
    Iterates through all files for one pass, shuffling each file's data,
    then moves to the next pass, up to max_passes times.
    """
    def __init__(self, data_dir, dataset, format='cot', max_passes=5,
                 batch_size=12, block_size=1024):
        self.format = format
        self.data_dir = os.path.join(data_dir, dataset)
        self.max_passes = max_passes
        self.batch_size = batch_size
        self.block_size = block_size
        
        # Collect all relevant files
        self.train_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.startswith('pencil' if self.format == 'pencil' else 'train')
            and f.endswith('.bin')
        ])
        if not self.train_files:
            raise ValueError(f"No training files found in {self.data_dir}")

        # Bookkeeping
        self.pass_idx = 0      # which pass are we on?
        self.file_idx = 0      # which file in the current pass?
        self.current_data = [] # data loaded from current file
        self.data_idx = 0      # index within current_data

        # load the first file
        self._load_current_file()

    def _load_current_file(self):
        """Loads and shuffles the current file."""
        if self.pass_idx >= self.max_passes:
            # we've exceeded the total passes allowed
            self.current_data = []
            return

        filename = self.train_files[self.file_idx]
        full_path = os.path.join(self.data_dir, filename)
        print(f"Loading file {filename} (pass {self.pass_idx+1}/{self.max_passes})...")
        with open(full_path, 'rb') as f:
            self.current_data = pickle.load(f)

        # Shuffle the data for this file
        random.shuffle(self.current_data)
        self.data_idx = 0

    def get_batch(self):
        """
        Returns (x, y) or (None, None) if we've finished all passes.
        """
        # If we finished all passes, stop
        if self.pass_idx >= self.max_passes:
            return None, None

        # If we've exhausted the current file, move to the next file
        if self.data_idx >= len(self.current_data):
            self.file_idx += 1
            # if we've used all files in one pass, increment pass_idx
            if self.file_idx >= len(self.train_files):
                self.file_idx = 0
                self.pass_idx += 1
                if self.pass_idx >= self.max_passes:
                    return None, None
            # load next file
            self._load_current_file()
            if not self.current_data:  # in case file is empty, or passes are done
                return None, None

        # gather up to batch_size indices from current_data
        end_idx = min(self.data_idx + self.batch_size, len(self.current_data))
        batch_indices = list(range(self.data_idx, end_idx))
        self.data_idx = end_idx

        if len(batch_indices) == 0:
            return None, None

        # build x, y depending on format
        if self.format == 'pencil':
            x, y = get_batch_pencil(batch_indices, self.current_data, self.block_size)
        else:
            x, y = get_batch_sequence(batch_indices, self.current_data, self.block_size)

        # move to device
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

        return x, y

# -----------------------------------------------------------------------------
# Batch builders
# -----------------------------------------------------------------------------
def get_batch_sequence(batch_indices, data, block_size):
    # Here, 'data' is a list of sequences. We just pick the ones in batch_indices.
    sequences = []
    for idx in batch_indices:
        seq = data[idx]
        if len(seq) > block_size + 1:
            seq = seq[:block_size + 1]
        sequences.append(seq)

    max_len = max(len(seq) for seq in sequences)
    x = torch.zeros((len(sequences), max_len - 1), dtype=torch.long)
    y = torch.zeros_like(x)

    # We'll need meta['stoi'] for the special prompt token
    # so we must ensure 'meta' is globally accessible or passed in.
    # For simplicity, we do the same trick as before:
    prompt_token_id = meta['stoi']['<|endofprompt|>']

    for i, seq in enumerate(sequences):
        x[i, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
        y[i, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)

        # zero out the loss up to the prompt token
        prompt_pos = (x[i] == prompt_token_id).nonzero()
        if len(prompt_pos) > 0:
            # zero out everything before the first prompt_pos
            y[i, :prompt_pos[0]] = 0

    return x, y


def get_batch_pencil(batch_indices, data, block_size):
    # 'data' is a list, where each element is itself a list of subinstances
    # We flatten subinstances from all selected items:
    subinstances = []
    for idx in batch_indices:
        subinstances.extend(data[idx])

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
        # zero out loss before mask_idx
        y[i, :mask_indices[i]] = 0

    return x, y

# -----------------------------------------------------------------------------
# Load validation data and tokenizer
# -----------------------------------------------------------------------------
# We'll load val_data and meta globally so we can access meta['stoi'] in get_batch_sequence()
data_dir_full = os.path.join(data_dir, dataset)
with open(os.path.join(data_dir_full, 'val.bin'), 'rb') as f:
    val_data = pickle.load(f)
with open(os.path.join(data_dir_full, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)    
tokenizer = SimpleTokenizer().load_from_meta(meta)

# -----------------------------------------------------------------------------
# Initialize the round-robin data loader
# -----------------------------------------------------------------------------
train_loader = RoundRobinDataLoader(
    data_dir=data_dir,
    dataset=dataset,
    format=format,
    max_passes=max_passes,
    batch_size=batch_size,
    block_size=block_size
)
print(f"Initialized round-robin DataLoader with {len(train_loader.train_files)} files.")

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
# A few auxiliary routines
# -----------------------------------------------------------------------------
def extract_label_ein(generated_text):
    """
    Example function used in evaluation. 
    """
    words = generated_text.split()
    for word in reversed(words):
        if word.strip() in ["Brit", "Swede", "Dane", "Norwegian", "German"]:
            return word.strip()
    return None

def evaluate_model_ein(model, val_data, tokenizer, ctx=None, format='pencil', 
                       max_new_tokens=1000, temperature=0.8, top_k=200,
                       num_samples=1000, log_info="noinfo", log_file="evaluation_log.txt",
                       out_dir="out", trace_rate=False, progress=None, training_loss=None):
    """Example evaluation routine."""
    num_correct = num_valid = 0
    device = next(model.parameters()).device
    average_trace_rate = []
    generation_times = []
    
    for idx, sample_tokens in enumerate(val_data[:num_samples]):
        # Process sample text
        sample_text = tokenizer.decode(sample_tokens)
        prompt = extract_prompt(sample_text)
        true_label = extract_label_ein(sample_text)

        # Generate prediction and measure time
        x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            with ctx or nullcontext():
                start_time = time.time()
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                                   tokenizer=tokenizer if format == 'pencil' else None,
                                   return_trace=trace_rate)
                end_time = time.time()
                generation_times.append(end_time - start_time)
                
        # Evaluate prediction
        pred_text = tokenizer.decode(y[0].tolist())
        pred_label = extract_label_ein(pred_text)
        
        print('------------------------------')
        print(f"Sample {idx + 1}")
        print(f"Prompt: {prompt}")
        print(f"Predict label: {pred_label}")
        print(f"True label: {true_label}")
        
        if pred_label:
            num_valid += 1
            if pred_label == true_label:
                num_correct += 1
        
        # Trace rate
        if trace_rate:
            rate = compute_trace_rate(sample_text.split(), pred_text.split())
            average_trace_rate.append(rate)
            
    print('------------------------------')
    # Calculate metrics
    accuracy = num_correct / num_samples if num_valid > 0 else 0
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
    print(f"Accuracy: {accuracy:.1%} ({num_correct}/{num_valid} correct)")
    print(f"Average generation time: {avg_time:.3f} seconds")
    if training_loss is not None:
        print(f"Current training loss: {training_loss:.5f}")
    
    # Calculate trace rate if enabled
    avg_trace = None
    if trace_rate and average_trace_rate:
        avg_trace = sum(average_trace_rate) / len(average_trace_rate)
        print(f"Average Trace Rate: {avg_trace:.1%}")
    
    # Log results
    if out_dir:
        with open(os.path.join(out_dir, log_file), 'a') as f:
            f.write(f"{log_info}\nAccuracy: {accuracy:.1%}\n")
            f.write(f"Average generation time: {avg_time:.3f} seconds\n")
            if training_loss is not None:
                f.write(f"Training loss: {training_loss:.5f}\n")
            if avg_trace is not None:
                f.write(f"Average Trace Rate: {avg_trace:.2%}\n")
            if progress is not None:
                f.write(f"Progress: {progress}\n")
            f.write("\n")
    
    return (accuracy, avg_trace) if trace_rate else accuracy

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
# prime the first batch
X, Y = train_loader.get_batch()
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

iter_num = 0
total_flops = 0
best_val_loss = 1e9
training_losses = []
last_eval_step = 0

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
        X, Y = train_loader.get_batch()
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
            training_loss=avg_loss
        )
        results = evaluate_model_ein(**args)

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
        progress=last_eval_step
    )
    results = evaluate_model_ein(**args)
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