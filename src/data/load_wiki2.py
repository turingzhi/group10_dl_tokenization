from datasets import load_dataset

def load_wiki2():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train = ds["train"]
    val   = ds["validation"]
    test  = ds["test"]

    # ↓↓↓ DEBUG / PRACTICAL SIZE ↓↓↓
    # train = train.select(range(1000))   # 10,000 articles
    val   = val #.select(range(1000))      # 1000 articles
    test  = test #.select(range(1000))     # 1000 articles

    return train, val, test