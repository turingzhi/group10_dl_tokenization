import torch
from torch.utils.data import Dataset
from typing import List, Union

class LMTokenizedDataset(Dataset):
    """
    Dataset class that converts a list of raw text segments into 
    context/target pairs (X, Y) for Causal Language Modeling training.
    
    It assumes the 'texts' list contains raw strings that need to be encoded.
    """
    def __init__(self, 
                 texts: List[str], # Should be a list of strings (ideally one long string)
                 tokenizer: object, 
                 context_length: int, 
                 stride: int = 16):

        # --- 1. Single, Clean Tokenization Pass ---
        # Initialize an empty list for all token IDs
        ids: List[int] = [] 
        
        # Iterate over the input list of text segments (which should contain 
        # just ONE segment: the entire corpus).
        for t in texts:
            # Call the tokenizer's encode method. This is the only place it should happen.
            # We assume ALL tokenizer wrappers (Byte, BPE, Unigram, etc.) have this method.
            # add_special_tokens=True ensures BOS/EOS are added to the sequence boundaries.
            ids.extend(tokenizer.encode(t, add_special_tokens=True))
            
        # --- 2. Final Tensor Assignment and Sequence Parameters ---
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride

        # Calculate the number of sequences. We need context_length tokens for X 
        # and one additional token (the target for the last X position) for Y.
        self.num_sequences = (len(self.ids) - context_length) // self.stride

    def __len__(self) -> int:
        """Returns the number of training sequences (X, Y pairs)."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts one (X, Y) sequence pair for Causal Language Modeling.
        """
        # Calculate start and end indices for the current sequence
        start = idx * self.stride
        end = start + self.context_length

        # X is the input sequence of length context_length
        x = self.ids[start:end]
        
        # Y is the target sequence, shifted by one token (next token prediction)
        y = self.ids[start+1:end+1] 
        
        return x, y