from torch.utils.data import DataLoader
from .tokenized_dataset import LMTokenizedDataset

def make_dataloaders(train, val, tokenizer, config):
    train_texts = [ex["text"] for ex in train]
    val_texts = [ex["text"] for ex in val]

    train_texts = [" ".join(train_texts)] 
    val_texts   = [" ".join(val_texts)]

    train_ds = LMTokenizedDataset(train_texts, tokenizer, config["context_length"], stride=16)
    val_ds   = LMTokenizedDataset(val_texts, tokenizer, config["context_length"], stride=16)

    return (
        DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True),
        DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False),
        train_ds,
        val_ds
    )
