from datasets import load_dataset
from pathlib import Path

def build_corpus(out_path="data/tokenizers/corpus.txt", split="train"):
    #ds = load_dataset("roneneldan/TinyStories", split=split)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex["text"].strip()
            if not text:
                continue
            f.write(text.replace("\n", " ") + "\n")

if __name__ == "__main__":
    build_corpus()
