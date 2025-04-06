import json
import pickle
import os

# Special tokens for prompt completion
SPECIAL_TOKENS = [
    '<|startoftext|>', # BOS
    '<|endofprompt|>', # EOP
    '<|endoftext|>' # EOS
]

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {'<UNK>': 0}  # Special token for unknown words
        self.idx2word = {0: '<UNK>'}
        self.vocab_size = 1
        
    def split_text(self, text):
        """Split text into tokens using whitespace and \n, keeping \n as a token."""
        tokens = []
        for part in text.split('\n'):
            if part:
                tokens.extend(part.split())
            tokens.append('\n')
        if tokens and tokens[-1] == '\n':
            tokens.pop()
            
        return tokens
    
    def build_vocab_from_file(self, data_dir, special_tokens=SPECIAL_TOKENS):
        """Build vocabulary from JSONL file.
        Args:
            data_dir: Directory containing JSONL file
            special_tokens: List of special tokens to add to vocab
        """
        # Build vocab from file
        filename = os.path.join(data_dir, 'data.jsonl')
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                
                # Split text into tokens using whitespace and \n, keeping \n as a token
                tokens = self.split_text(text)
                    
                for token in tokens:
                    if token not in self.word2idx:
                        self.word2idx[token] = self.vocab_size
                        self.idx2word[self.vocab_size] = token
                        self.vocab_size += 1

        # Add special tokens
        if special_tokens:
            for token in special_tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1
                print(f"Added special token {token} with index {self.word2idx[token]}")

        # Print mapping of tokens to IDs
        print("\nToken to ID mapping:")
        for token, idx in self.word2idx.items():
            print(f"{token}: {idx}")

        return self
    
    def load_from_meta(self, meta):
        """Load meta information to rebuild tokenizer."""
        self.vocab_size = meta['vocab_size'] 
        self.idx2word = meta['itos']
        self.word2idx = meta['stoi']
        return self

    def encode(self, text, with_eos=False, with_bos=False):
        """Convert text to token IDs. Handling out-of-vocabulary (OOV) tokens with a default value (0)."""
        tokens = self.split_text(text)
        
        if with_eos:
            tokens.append('<|endoftext|>')
        if with_bos:
            tokens.insert(0, '<|startoftext|>')
        return [self.word2idx.get(token, 0) for token in tokens]
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return ' '.join(self.idx2word[idx] for idx in token_ids)

    def get_vocab_size(self):
        """Return vocabulary size."""
        return self.vocab_size

    def save_meta_to_file(self, data_dir):
        """Save meta information to a file."""
        meta = {
            'vocab_size': self.vocab_size,
            'itos': self.idx2word,
            'stoi': self.word2idx,
        }
        filename = os.path.join(data_dir, 'meta.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(meta, f)
