from pathlib import Path
from tokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers, processors, decoders

# --- Configuration (Keeping these global) ---
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
MAX_SEQUENCE_LENGTH = 512 # Assuming this is your max sequence length
TOKENIZER_SAVE_DIR = "data/tokenizers/unigram"

class UnigramTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

        # Initialization is fine
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

    @classmethod
    def train(cls, corpus_path, vocab_size=5000, save_dir="data/tokenizers/unigram"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = Tokenizer(models.Unigram())

        # 1. FIX: Use Metaspace Pre-Tokenizer for proper subword segmentation
        # Metaspace replaces spaces with a special character (e.g., ' ') and tokenizes based on that.
        # This allows the Unigram model to learn subwords spanning across word boundaries.
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace() 

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            unk_token="<unk>",
            # Note: UnigramTrainer often benefits from an initial alphabet like BPE, but
            # relying on the Metaspace pre-tokenizer is usually sufficient for English corpora.
        )

        print(f"Starting Unigram training with corpus: {corpus_path}")
        tokenizer.train(files=[str(corpus_path)], trainer=trainer)
        print("Training complete.")
        
        # 2. FIX: Set Metaspace Decoder (Essential for reversible tokenization)
        tokenizer.decoder = decoders.Metaspace()

        # 3. FIX: Set Post-Processor for BOS/EOS and padding/truncation
        # This automates the special token handling cleanly.
        bos_id = tokenizer.token_to_id(SPECIAL_TOKENS[2])
        eos_id = tokenizer.token_to_id(SPECIAL_TOKENS[3])
        
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{SPECIAL_TOKENS[2]} $A {SPECIAL_TOKENS[3]}",
            pair=f"{SPECIAL_TOKENS[2]} $A {SPECIAL_TOKENS[3]} $B {SPECIAL_TOKENS[3]}",
            special_tokens=[
                (SPECIAL_TOKENS[2], bos_id),
                (SPECIAL_TOKENS[3], eos_id),
            ],
        )
        
        tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        tokenizer.enable_padding(
            direction="right",
            pad_id=tokenizer.token_to_id("<pad>"),
            pad_token="<pad>"
        )


        tokenizer.save(str(save_dir / "tokenizer.json"))
        return cls(tokenizer)

    # 4. FIX: Simplify encode/decode to use the configured pipeline
    def encode(self, text, add_special_tokens=True):
        # The Post-Processor handles adding BOS/EOS via the `add_special_tokens` argument
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids, skip_special_tokens=True):
        # The Metaspace Decoder handles stripping the hidden space markers
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @classmethod
    def load(cls, save_dir=TOKENIZER_SAVE_DIR):
        """
        Loads the saved tokenizer from a file.
        """
        tokenizer_path = Path(save_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return cls(tokenizer)

def main():
    # --- Tokenization Test Setup (Insert this section before the test block) ---

# 1. Define corpus path (replace with your actual path)
    corpus_path = "data/corpus.txt" 
# NOTE: Ensure the file at this path exists for training to work.

# 2. Train and instantiate the tokenizer object
# This calls the @classmethod train(), which returns an instance of UnigramTokenizer
    tokenizer_wrapper = UnigramTokenizer.train(corpus_path=corpus_path, vocab_size=5000)

# --- Tokenization Test ---
    test_text = " The quick brown fox jumps over the lazy dog. " 

# FIX: Call methods on the instantiated object, 'tokenizer_wrapper'
    encoded_ids = tokenizer_wrapper.encode(test_text, add_special_tokens=False) 
    decoded_text = tokenizer_wrapper.decode(encoded_ids, skip_special_tokens=False) # Use False to check internal special tokens

    print(f"Original: '{test_text}'")
    print(f"Decoded: '{decoded_text}'")
    print(f"Encoded IDs: {encoded_ids}")

# Final success check: strip special tokens if the decode method doesn't
    if decoded_text.strip() == test_text.strip():
        print("✅ Reversibility SUCCESSFUL.")
    else:
    # Adding extra info to debug failed test
        print(f"❌ Reversibility FAILED. Original length: {len(test_text)}, Decoded length: {len(decoded_text)}")


if __name__ == "__main__":
    main()