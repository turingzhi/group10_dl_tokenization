import json
from collections import Counter
from pathlib import Path
import re
from typing import List, Dict

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

class WordTokenizer:
    def __init__(self, stoi: Dict[str, int], itos: List[str]):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(itos)

        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """Split text into tokens using simple alphanumeric segmentation."""
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    @classmethod
    def train(cls, corpus_path: str, vocab_size: int = 5000, save_dir: str = "data/tokenizers/word"):
        """Train a word-level tokenizer and save the vocabulary."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        counter = Counter()

        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = cls.tokenize_text(line)
                if tokens:
                    counter.update(tokens)

        max_words = vocab_size - len(SPECIAL_TOKENS)

        # deterministic ordering
        sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        most_common = [w for w, _ in sorted_words[:max_words]]

        itos = SPECIAL_TOKENS + most_common
        stoi = {word: idx for idx, word in enumerate(itos)}

        with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(itos, f, ensure_ascii=False, indent=2)

        return cls(stoi, itos)

    @classmethod
    def load(cls, save_dir: str = "data/tokenizers/word"):
        """Load tokenizer from saved vocabulary."""
        save_dir = Path(save_dir)
        with open(save_dir / "vocab.json", encoding="utf-8") as f:
            itos = json.load(f)
        stoi = {word: idx for idx, word in enumerate(itos)}
        return cls(stoi, itos)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text into token IDs."""
        tokens = self.tokenize_text(text)
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokens]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back into a sentence."""
        words = []
        for idx in ids:
            if idx < 0 or idx >= self.vocab_size:
                continue
            tok = self.itos[idx]
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            words.append(tok)
        return " ".join(words)