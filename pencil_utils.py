import os
import numpy as np
import pickle
import json
import torch

PENCIL_TOKENS = {
    'call': '[CALL]',
    'sep': '[SEP]',
    'return': '[RETURN]'
}

##########################################
# Truncate the sequence at the first return token:
# A [CALL] B [SEP] C [RETURN] D => A [CALL] B [SEP] C [RETURN]
##########################################
def truncate_encoding(encoding, return_token):
    # Find the first return token
    return_index = np.where(encoding == return_token)[0][0]
    # Use numpy array slicing
    segment = encoding[:return_index + 1]
    return segment


##########################################
# Apply the reduction rule:
# Reduce the token sequence by removing the part between [SEP] and [CALL]
# including special token triplet themselves: [CALL] B [SEP] C [RETURN] => C
##########################################
'''For numpy arrays used for generating the dataset'''
def reduce_encoding(encoding, sep_token, call_token, return_token):
    # Store initial encoding
    initial_encoding = encoding.clone() if isinstance(encoding, torch.Tensor) else encoding.copy()
    
    # Convert torch tensor to numpy if needed
    is_torch = False
    if isinstance(encoding, torch.Tensor):
        is_torch = True
        device = encoding.device
        dtype = encoding.dtype
        encoding = encoding.squeeze().cpu().numpy()
    try:
        # Find the first return token
        return_indices = np.where(encoding == return_token)[0]
        if len(return_indices) == 0:
            print("Error in encoding")
            return initial_encoding
        next_func = return_indices[0]
        
        # Find the last sep token before the return token
        sep_indices_before = np.where(encoding[:next_func] == sep_token)[0]
        if len(sep_indices_before) == 0:
            print("Error in encoding")
            return initial_encoding
        sep_index = sep_indices_before[-1]
        
        # Find the last call token before the sep token
        func_tokens_before = np.where(encoding[:sep_index] == call_token)[0]
        if len(func_tokens_before) == 0:
            print("Error in encoding")
            return initial_encoding
        last_func = func_tokens_before[-1]
    except:
        print("Error in encoding")
        return initial_encoding
    
    # Concatenate arrays using numpy.concatenate
    reduced_encoding = np.concatenate([
        encoding[:last_func],
        encoding[sep_index + 1: next_func],
        encoding[next_func + 1:]
    ])
    
    # Convert back to torch tensor if input was torch tensor
    if is_torch:
        reduced_encoding = torch.tensor(reduced_encoding, device=device, dtype=dtype).unsqueeze(dim=0)
        # print(reduced_encoding.shape)
    
    return reduced_encoding

'''For torch tensors used in the model'''
def reduce_encoding_tensor(encoding, sep_token, call_token, return_token):
    # Make a copy of the original encoding
    initial_encoding = encoding.clone()

    # Handle potential batch dimension (assume shape [B, seq_len] or [seq_len])
    squeeze_needed = (encoding.dim() == 2 and encoding.size(0) == 1)
    if squeeze_needed:
        encoding = encoding.squeeze(0)
    elif encoding.dim() > 1:
        # If there's more than 2 dimensions or a multi-batch scenario,
        # you may need additional logic. For now, we assume single-batch or no batch.
        return initial_encoding  # or raise an error

    # Find the first return token
    return_indices = torch.nonzero(encoding == return_token, as_tuple=True)[0]
    if return_indices.numel() == 0:
        # No return token found, return the original
        return initial_encoding
    first_return_idx = return_indices[0].item()
    
    # Find the last sep token before the return token
    sep_indices_before = torch.nonzero(encoding[:first_return_idx] == sep_token, as_tuple=True)[0]
    if sep_indices_before.numel() == 0:
        # No separator found before return, return the original
        return initial_encoding
    sep_index = sep_indices_before[-1].item()
    
    # Find the last call token before the sep token
    call_indices_before = torch.nonzero(encoding[:sep_index] == call_token, as_tuple=True)[0]
    if call_indices_before.numel() == 0:
        # No call token found before the separator
        return initial_encoding
    last_call_idx = call_indices_before[-1].item()

    # Rebuild the reduced encoding
    # Part 1: up to (but not including) the last_call_idx
    # Part 2: from just after sep_index to just before first_return_idx
    # Part 3: everything after first_return_idx
    part1 = encoding[:last_call_idx]
    part2 = encoding[sep_index + 1:first_return_idx]
    part3 = encoding[first_return_idx + 1:]

    reduced_encoding = torch.cat([part1, part2, part3], dim=0)

    # Re-insert the batch dimension if necessary
    if squeeze_needed:
        reduced_encoding = reduced_encoding.unsqueeze(0)

    return reduced_encoding


##########################################
# Convert a very long sequence into a set of subsequences for training pencil
##########################################
def extract_subseqs(encoding, sep_token, call_token, return_token, eop_token):
    subseqs = []
    
    # Find initial mask_idx
    mask_idx = np.where(encoding == eop_token)[0][0]
    
    # Continue while separator token exists in encoding
    while sep_token in encoding:
        # Truncate encoding at function return token and include it in subseqs
        truncated_encoding = truncate_encoding(encoding, return_token)
        subseqs.append({'ids': truncated_encoding, 'mask_idx': mask_idx})
        
        # Find separator token index
        sep_index = np.where(encoding == sep_token)[0][0]
        
        # Find last function token before separator
        func_tokens_before = np.where(encoding[:sep_index] == call_token)[0]
        last_func = func_tokens_before[-1]
        
        # Find next function token after separator
        func_tokens_after = np.where(encoding[sep_index + 1:] == return_token)[0]
        next_func = func_tokens_after[0] + sep_index + 1    
        
        # Compute the mask index for the next iteration
        mask_idx = last_func + (next_func - sep_index) - 2
        
        # Reduce encoding by the transition rule [CALL] B [SEP] C [RETURN] => C
        encoding = np.concatenate([
            encoding[:last_func],
            encoding[sep_index + 1: next_func],
            encoding[next_func + 1:]
        ])
    
    subseqs.append({'ids': encoding, 'mask_idx': mask_idx})
    
    return subseqs


##########################################
# Pipeline for processing the dataset
##########################################
def calculate_flops(x_sequence, y_sequence, sep_token, call_token, return_token, format='pencil'):
    flops = 0
    
    # Get positions of non-zero elements using a for loop
    # seq_indices is the index of non-zero elements (tokens that need to compute the loss)
    batch_indices, seq_indices = torch.nonzero(y_sequence, as_tuple=True)
    
    # Use for loop to add 1 to indices and calculate the sum
    for index in seq_indices:
        flops += int(index.item() + 1)

    if format == 'pencil': # account for the reduction
        for i in range(len(x_sequence)):
            # Find the first separator token index
            sep_indices = torch.nonzero(x_sequence[i] == sep_token, as_tuple=True)[0]
            if sep_indices.numel() == 0:
                continue
            sep_index = sep_indices[0].item()

            # Find all call tokens before separator, then pick the last one
            call_indices_before = torch.nonzero(x_sequence[i][:sep_index] == call_token, as_tuple=True)[0]
            if call_indices_before.numel() == 0:
                continue
            call_index = call_indices_before[-1].item()

            # Find the first return token
            return_indices_after = torch.nonzero(y_sequence[i] == return_token, as_tuple=True)[0]
            if return_indices_after.numel() == 0:
                # No return token found after the separator
                continue
            return_index = return_indices_after[0].item() + 1 # add 1 because of the offset 
        
            for token_index in range(sep_index + 1, return_index):
                flops += token_index - (sep_index - call_index + 1)
                    
    return flops


##########################################
# Pipeline for tokenizing and spliting the dataset
##########################################
def process_dataset(data_dir, train_size=100000, val_size=1000, test_size=1000, tokenizer=None):
    """
    Process large JSONL dataset with batched training data.
    """
    # Remove existing train files
    for file in os.listdir(data_dir):
        if file.startswith('train') and file.endswith('.bin'):
            os.remove(os.path.join(data_dir, file))
            print(f"Removed existing file: {file}")

    def tokenize_sequences(texts):
        return [np.array(tokenizer.encode(text, with_eos=True, with_bos=True), 
                        dtype=np.uint8) for text in texts]
    
    def print_stats(name, sequences):
        num_tokens = sum(len(seq) for seq in sequences)
        max_len = max(len(seq) for seq in sequences)
        print(f"{name} set: {len(sequences):,} sequences, {num_tokens:,} tokens")
        print(f"{name} max sequence length: {max_len}")
    
    jsonl_path = os.path.join(data_dir, 'data.jsonl')
    
    # Process validation and test sets first
    print("Processing validation and test sets...")
    val_data = []
    test_data = []
    train_data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        # Read first val_size + test_size samples for val/test sets
        for i, line in enumerate(f):
            if i < val_size:
                val_data.append(json.loads(line)['text'])
            elif i < val_size + test_size:
                test_data.append(json.loads(line)['text'])
            else:
                break
    
    # Tokenize and save val/test sets
    print("Tokenizing validation and test sets...")
    val_ids = tokenize_sequences(val_data)
    test_ids = tokenize_sequences(test_data)
    
    print_stats("Validation", val_ids)
    print_stats("Test", test_ids)
    
    # Save val/test sets
    for name, data in [("val", val_ids), ("test", test_ids)]:
        save_path = os.path.join(data_dir, f'{name}.bin')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    # Process training data in batches
    print("Processing training data in batches...")
    batch_num = 0
    current_batch = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        # Skip val/test samples
        for _ in range(val_size + test_size):
            next(f)
            
        # Process remaining lines in batches
        for i, line in enumerate(f):
            current_batch.append(json.loads(line)['text'])
            
            if len(current_batch) == train_size:
                print(f"Processing training batch {batch_num + 1}...")
                batch_ids = tokenize_sequences(current_batch)
                
                save_path = os.path.join(data_dir, f'train{batch_num}.bin')
                with open(save_path, 'wb') as bf:
                    pickle.dump(batch_ids, bf)
                
                print_stats(f"Training batch {batch_num + 1}", batch_ids)
                current_batch = []
                batch_num += 1
    
    # Save final partial batch if any
    if current_batch:
        print(f"Processing final training batch...")
        batch_ids = tokenize_sequences(current_batch)
        save_path = os.path.join(data_dir, f'train{batch_num}.bin')
        with open(save_path, 'wb') as bf:
            pickle.dump(batch_ids, bf)
        print_stats(f"Training batch {batch_num + 1}", batch_ids)
    
    print(f"Saved {batch_num + 1} training batches to {data_dir}/train[0-{batch_num}].bin")


##########################################
# Pipeline for converting training batches to pencil format
##########################################
def process_pencil_data(data_dir, tokenizer):
    """Process multiple training batches into pencil format."""
    print("Converting data to pencil format...")
    
    # Remove existing pencil files
    for file in os.listdir(data_dir):
        if file.startswith('pencil') and file.endswith('.bin'):
            os.remove(os.path.join(data_dir, file))
            print(f"Removed existing file: {file}")

    # Find all training batch files 
    train_files = sorted([f for f in os.listdir(data_dir) if f.startswith('train') and f.endswith('.bin')])
    
    total_samples = 0
    total_subseqs = 0
    total_tokens_before = 0 
    total_tokens_after = 0
    max_len_before = 0
    max_len_after = 0
    
    for train_file in train_files:
        batch_num = int(train_file.split('train')[1].split('.')[0])  # Extract N from 'trainN.bin'
        print(f"\nProcessing batch {batch_num}...")
        
        # Load and process batch
        # the data is a list of list, each list represents a sample, each item in the list is a subsequence
        all_subseqs = []
        train_path = os.path.join(data_dir, train_file)
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        
        # Process each sample
        for encoding in train_data:
            subseqs = extract_subseqs(
                encoding,
                tokenizer.word2idx[PENCIL_TOKENS['sep']],
                tokenizer.word2idx[PENCIL_TOKENS['call']],
                tokenizer.word2idx[PENCIL_TOKENS['return']],
                tokenizer.word2idx['<|endofprompt|>']
            )
            all_subseqs.append(subseqs)
        
        # Save pencil batch
        pencil_file = f'pencil{batch_num}.bin'
        pencil_path = os.path.join(data_dir, pencil_file)
        with open(pencil_path, 'wb') as f:
            pickle.dump(all_subseqs, f)
        
        # Collect statistics
        batch_tokens_before = sum(len(sample) for sample in train_data)
        max_len_before = max(max_len_before, max(len(sample) for sample in train_data))
        total_tokens_before += batch_tokens_before
        total_samples += len(train_data)
        
        flattened = [subsample for sample in all_subseqs for subsample in sample]
        batch_tokens_after = sum(len(sample['ids']) for sample in flattened)
        max_len_after = max(max_len_after, max(len(sample['ids']) for sample in flattened))
        total_tokens_after += batch_tokens_after
        total_subseqs += len(flattened)
        
        print(f"Batch {batch_num} stats:")
        print(f"Sequences: {len(train_data)} → {len(flattened)}")
    
    print("\nOverall statistics:")
    print(f"Total sequences: {total_samples} → {total_subseqs}")
    print(f"Max length: {max_len_before} → {max_len_after}")
    print(f"Average length: {total_tokens_before/total_samples:.1f} → {total_tokens_after/total_subseqs:.1f}")
    print(f"Average reductions per sample: {total_subseqs/total_samples:.1f}")
    
    return total_subseqs