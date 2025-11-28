from pathlib import Path
from tokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

class UnigramTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

    @classmethod
    def train(cls, corpus_path, vocab_size=5000, save_dir="data/tokenizers/unigram"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = Tokenizer(models.Unigram())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            unk_token="<unk>",
        )

        tokenizer.train(files=[corpus_path], trainer=trainer)

        tokenizer.save(str(save_dir / "tokenizer.json"))
        return cls(tokenizer)

    @classmethod
    def load(cls, save_dir="data/tokenizers/unigram"):
        tokenizer = Tokenizer.from_file(str(Path(save_dir) / "tokenizer.json"))
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