from pathlib import Path
from tokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers, processors, normalizers, decoders

# --- Configuration ---
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
MAX_SEQUENCE_LENGTH = 512
TOKENIZER_SAVE_DIR = "data/tokenizers/bpe"

class BPETokenizer:
    
    @classmethod
    def train(cls, corpus_path, vocab_size=15000, save_dir=TOKENIZER_SAVE_DIR):
        """
        Trains the ByteLevel BPE tokenizer and saves the configuration.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Byte-Level BPE model
        model = models.BPE(unk_token="<unk>")
        tokenizer = Tokenizer(model)

        # 2. Normalization and Pre-Tokenization
        # NFKC normalizes Unicode characters (e.g., combining characters)
        tokenizer.normalizer = normalizers.NFKC()
        # ByteLevel pre-tokenizer handles all bytes and typically adds a prefix space (crucial for decoding)
        bytelevel = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.pre_tokenizer = bytelevel

        # 3. Trainer configuration
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=bytelevel.alphabet(),
        )

        # 4. Train the tokenizer
        print(f"Starting BPE training with corpus: {corpus_path}")
        tokenizer.train([str(corpus_path)], trainer)
        print("Training complete.")

        # 4.5. Set the Decoder (CRUCIAL FIX)
        # This reverses the ByteLevel pre-tokenization, correctly handling whitespace markers.
        tokenizer.decoder = decoders.ByteLevel()

        # 5. Post-processing (Crucial for correct BOS/EOS handling and decoding)
        
        # Enable padding/truncation settings (important for model input pipeline)
        tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH) 
        tokenizer.enable_padding(
            direction="right", 
            pad_id=tokenizer.token_to_id("<pad>"), 
            pad_token="<pad>"
        ) 

        # The TemplateProcessing wraps the token IDs with special tokens automatically during encoding
        # This fixes the issue of manual token handling in the decode step.
        bos_id = tokenizer.token_to_id(SPECIAL_TOKENS[2])
        eos_id = tokenizer.token_to_id(SPECIAL_TOKENS[3])
        
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{SPECIAL_TOKENS[2]} $A {SPECIAL_TOKENS[3]}",  # <bos> $A <eos>
            pair=f"{SPECIAL_TOKENS[2]} $A {SPECIAL_TOKENS[3]} $B {SPECIAL_TOKENS[3]}",
            special_tokens=[
                (SPECIAL_TOKENS[2], bos_id),
                (SPECIAL_TOKENS[3], eos_id),
            ],
        )

        # 6. Save and return
        tokenizer.save(str(save_dir / "vocab.json"))
        print(f"Tokenizer saved to {save_dir / 'vocab.json'}")

        return cls(tokenizer)

    @classmethod
    def load(cls, save_dir=TOKENIZER_SAVE_DIR):
        """
        Loads the saved tokenizer from a file.
        """
        tokenizer_path = Path(save_dir) / "vocab.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return cls(tokenizer)

    def __init__(self, tokenizer):
        """
        Initializes the wrapper class with a loaded tokenizer.
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

        # Get IDs for convenience
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.unk_id = tokenizer.token_to_id("<unk>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    def encode(self, text, add_special_tokens=True):
        """
        Encodes text into a list of token IDs.
        Uses the Post Processor to handle BOS/EOS insertion.
        """
        # The library's encode method uses the configured Post Processor and Pre-tokenizer
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Decodes a list of token IDs back into text.
        The library's decode method correctly reverses the ByteLevel encoding.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

# --- Main execution script ---
def main():
    # Ensure this path exists and contains your raw training text
    corpus = "data/corpus.txt" 
    
    # 1. Create a dummy corpus file for a minimal reproducible test (optional)
    corpus_path = Path(corpus)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    if not corpus_path.exists():
         with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write("Hello world\n")
            f.write("This is a test of BPE tokenization.\n")
            f.write("The quick brown fox jumps over the lazy dog.\n")
            f.write("H ello world\n") # Include the problematic string

    # 2. Train and save the tokenizer
    tokenizer_wrapper = BPETokenizer.train(corpus_path=corpus, vocab_size=15000)

    # 3. Test Encoding and Decoding
    test_text = "H ello world" 

    # Encoding
    encoded_ids = tokenizer_wrapper.encode(test_text, add_special_tokens=True)
    
    # Decoding (This should now correctly handle the ByteLevel output)
    decoded_text = tokenizer_wrapper.decode(encoded_ids, skip_special_tokens=True)

    print("\n--- Tokenization Test ---")
    print(f"Original Text: '{test_text}'")
    print(f"Encoded IDs: {encoded_ids}")
    print(f"Decoded Text: '{decoded_text}'")
    
    if decoded_text.strip() == test_text.strip():
        print("✅ Decoding test SUCCESSFUL: Decoded text matches original text.")
    else:
        print("❌ Decoding test FAILED: Decoded text does NOT match original text.")

if __name__ == "__main__":
    main()