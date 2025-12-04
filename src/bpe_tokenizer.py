from pathlib import Path
from tokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.unk_id = tokenizer.token_to_id("<unk>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    @classmethod
    def train(cls, corpus_path, vocab_size=5000, save_dir="data/tokenizers/bpe"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
        )

        tokenizer.train(files=[corpus_path], trainer=trainer)
        # tokenizer.save(str(save_dir / "tokenizer.json"))
        tokenizer.model.save(str(save_dir))              # saves vocab.json + merges.txt
        tokenizer.save(str(save_dir / "tokenizer.json")) # saves full config

        return cls(tokenizer)

    @classmethod
    def load(cls, save_dir="data/tokenizers/bpe"):
        # tokenizer = Tokenizer.from_file(str(Path(save_dir) / "tokenizer.json"))
        tokenizer = Tokenizer(models.BPE.from_file(
            vocab=str(Path(save_dir) / "vocab.json"),
            merges=str(Path(save_dir) / "merges.txt"),
            unk_token="<unk>"
        ))
        return cls(tokenizer)

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            text = "<bos> " + text + " <eos>"
        return self.tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=True):
        text = self.tokenizer.decode(ids)
        if skip_special_tokens:
            for t in SPECIAL_TOKENS:
                text = text.replace(t, "").strip()
        return text