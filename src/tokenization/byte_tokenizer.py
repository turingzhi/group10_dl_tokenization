# --- src/tokenization/byte_tokenizer.py ---

# Define special tokens globally
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

class ByteTokenizer:
    """
    A custom tokenizer wrapper that treats the raw UTF-8 bytes of text 
    as tokens. Its vocabulary is fixed (4 special tokens + 256 bytes).
    """
    def __init__(self):
        # 1. Define the fixed vocabulary size and special token indices
        self.num_special = len(SPECIAL_TOKENS)
        self.vocab_size = self.num_special + 256  # Total: 260
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    @classmethod
    def load(cls, save_dir=None):
        """
        Loads or initializes the tokenizer. Since byte tokenizers are fixed, 
        this simply returns a new instance.
        """
        # Kept for API consistency with BPE/Unigram.
        return cls()

    def encode(self, text, add_special_tokens=True):
        """
        Converts a raw string into a list of token IDs.
        
        The core logic: byte value (0-255) is offset by self.num_special (4) 
        to yield token ID (4-259).
        """
        # Convert text to UTF-8 bytes (this is where multi-byte characters become 2, 3, or 4 bytes)
        byte_ids = list(text.encode("utf-8")) 
        
        # Shift byte values to token ID space (e.g., byte 0 -> ID 4)
        ids = [b + self.num_special for b in byte_ids]
        
        # Add special tokens
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Converts a list of token IDs back into a raw string.
        """
        bytes_list = []
        for i in ids:
            if i < self.num_special:
                # Handle special tokens
                if skip_special_tokens:
                    continue
                # If not skipping, you could map special IDs to a display character
            else:
                # Revert shift: Token ID (4-259) -> byte value (0-255)
                bytes_list.append(i - self.num_special)
                
        # Convert the list of byte values back into a UTF-8 string
        return bytes(bytes_list).decode("utf-8", errors="ignore")