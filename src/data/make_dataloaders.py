from torch.utils.data import DataLoader
from .tokenized_dataset import LMTokenizedDataset

def make_dataloaders(train, val, test, tokenizer, config):
    train_texts = [ex["text"] for ex in train]
    val_texts = [ex["text"] for ex in val]
    test_texts = [ex["text"] for ex in test]

    train_texts = [" ".join(train_texts)] 
    val_texts   = [" ".join(val_texts)]
    test_texts   = [" ".join(test_texts)]

    train_ds = LMTokenizedDataset(train_texts, tokenizer, config["context_length"], stride=16)
    val_ds   = LMTokenizedDataset(val_texts, tokenizer, config["context_length"], stride=16)
    test_ds   = LMTokenizedDataset(test_texts, tokenizer, config["context_length"], stride=16)

    return (
        DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True),
        DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False),
        DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False),
        train_ds,
        val_ds,
        test_ds
    )
