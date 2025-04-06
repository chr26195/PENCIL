import os
import pickle
from contextlib import nullcontext
import torch
import time
from tokenizer import SimpleTokenizer
from model import GPTConfig, GPT
from pencil_utils import *


# Define functions to extract the prompt, ground truth label, and predicted label from a sample text
def extract_prompt(text):
    """Extract the prompt part from a sample text."""
    if '<|endofprompt|>' not in text:
        return text
    return text.split('<|endofprompt|>')[0] + '<|endofprompt|>'

def extract_label(generated_text):
    """
    Extract label from generated text by finding the last True/False occurrence.
    Returns None if no True/False is found.
    """
    words = generated_text.split()
    for word in reversed(words):
        if word.strip() in ['True', 'False']:
            return word.strip()
    return None

def compute_trace_rate(list1, list2):
    # Handle empty lists
    if not list1 and not list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    
    # Count matching elements
    matches = sum(1 for i in range(min(len(list1), len(list2))) 
                 if list1[i] == list2[i])
    
    # Calculate trace rate
    trace_rate = matches / max(len(list1), len(list2))
    return trace_rate

def evaluate_model(model, val_data, tokenizer, ctx=None, format='pencil', max_new_tokens=1000, temperature=0.8, top_k=200, num_samples=1000, log_info="noinfo", log_file="evaluation_log.txt", out_dir="out", trace_rate=False, progress=None, training_loss=None, iter=None):
    """Evaluate model accuracy on validation data."""
    num_correct = num_valid = 0
    device = next(model.parameters()).device
    average_trace_rate = []
    generation_times = []
    
    for idx, sample_tokens in enumerate(val_data[:num_samples]):
        # Process sample text
        sample_text = tokenizer.decode(sample_tokens)
        prompt = extract_prompt(sample_text)
        true_label = extract_label(sample_text)

        # Generate prediction and measure time
        x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            with ctx or nullcontext():
                start_time = time.time()
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                                    tokenizer=tokenizer if format == 'pencil' else None, return_trace = trace_rate)
                end_time = time.time()
                generation_times.append(end_time - start_time)
                
        # Evaluate prediction
        pred_text = tokenizer.decode(y[0].tolist())
        pred_label = extract_label(pred_text)
        
        print('------------------------------')
        print(f"Sample {idx + 1}")
        print(f"Prompt: {prompt}")
        print(f"Generated: {pred_text}")
        print(f"True label: {true_label}")
        
        if pred_label:
            num_valid += 1
            if pred_label == true_label:
                num_correct += 1
        
        # Trace rate
        if trace_rate:
            # print(f"Label    : {sample_text}")
            # print(f"Generated: {pred_text}")
            trace_rate = compute_trace_rate(sample_text.split(), pred_text.split())
            average_trace_rate.append(trace_rate)
            
            
    print('------------------------------')
    # Calculate metrics
    accuracy = num_correct / num_samples if num_valid > 0 else 0
    avg_time = sum(generation_times) / len(generation_times)
    print(f"Accuracy: {accuracy:.1%} ({num_correct}/{num_valid} correct)")
    print(f"Average generation time: {avg_time:.3f} seconds")
    if training_loss is not None:
        print(f"Current training loss: {training_loss:.5f}")
    
    # Calculate trace rate if enabled
    avg_trace = None
    if trace_rate and average_trace_rate:
        avg_trace = sum(average_trace_rate) / len(average_trace_rate)
        print(f"Average Trace Rate: {avg_trace:.2%}")
    
    # Log results
    if out_dir:
        with open(os.path.join(out_dir, log_file), 'a') as f:
            f.write(f"{log_info}\nAccuracy: {accuracy:.1%}\n")
            f.write(f"Average generation time: {avg_time:.3f} seconds\n")
            if training_loss is not None:
                f.write(f"Training loss: {training_loss:.5f}\n")
            if avg_trace is not None:
                f.write(f"Average Trace Rate: {avg_trace:.2%}\n")
            if progress:
                f.write(f"Progress: {progress}\n")
            if iter:
                f.write(f"Iteration: {iter}\n")
            f.write("\n")
    
    return (accuracy, avg_trace) if trace_rate else accuracy

if __name__ == '__main__':
    format = 'pencil' # either 'pencil' or 'cot'
    log_info = "noinfo"
    log_file = "evaluation_log.txt"
    trace_rate = False

    # -----------------------------------------------------------------------------
    dataset = '3sat'
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out'  # ignored if init_from is not 'resume'
    num_samples = 1000  # number of samples to evaluate
    max_new_tokens = 1000 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    out_dir = os.path.join('out', dataset)

    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # Set up the environment
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load the model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)
        
    # Load the tokenizer    
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    if not os.path.exists(meta_path):
        raise ValueError("meta.pkl not found. Please provide meta.pkl file.")

    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    tokenizer = SimpleTokenizer()
    tokenizer.load_from_meta(meta)

    # Load the validation data
    data_dir = os.path.join('data', dataset)
    val_bin_path = os.path.join(data_dir, 'val.bin')
    with open(val_bin_path, 'rb') as f:
        val_data = pickle.load(f)  # List of NumPy arrays
    print(f"Loaded {len(val_data)} validation samples.")
    
    # Evaluate the model
    accurary = evaluate_model(
        model, val_data, tokenizer, ctx=ctx, format=format, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, num_samples=num_samples, log_info=log_info, out_dir=out_dir,
        log_file="evaluation_log.txt"
    )
    