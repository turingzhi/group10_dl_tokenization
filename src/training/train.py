import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.config import config
from src.data.load_tinystories import load_tinystories
from src.data.make_dataloaders import make_dataloaders
from src.tokenization.factory import get_tokenizer
from src.models.transformer_lm import TransformerLM
from src.models.lstm_lm import LSTMLM


def train_model(tokenizer_name: str, model_type: str):
    print(f"Starting training with tokenizer='{tokenizer_name}', model_type='{model_type}'")
    device = config["device"]

    # 1) tokenizer and data
    tokenizer = get_tokenizer(tokenizer_name)
    print("Loading TinyStories...")
    train_ds, val_ds = load_tinystories()
    print("Loaded TinyStories:", len(train_ds), "train stories,", len(val_ds), "val stories")

    print("Building tokenized datasets and dataloaders...")
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, tokenizer, config)
    print("Dataloaders ready. Starting epochs...")
    # train_ds, val_ds = load_tinystories()
    # train_loader, val_loader = make_dataloaders(train_ds, val_ds, tokenizer, config)

    vocab_size = tokenizer.vocab_size

    # 2) model
    if model_type == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            model_dim=config["model_dim"],
            n_heads=config["num_heads"],
            n_layers=config["num_layers"],
            context_length=config["context_length"],
        )
    elif model_type == "lstm":
        model = LSTMLM(vocab_size, config["model_dim"])
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    # 3) training loop
    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch} starting...")
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"{tokenizer_name}-{model_type} epoch {epoch}"):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        # (optional) save checkpoint
        out_path = f"checkpoints/{tokenizer_name}_{model_type}_epoch{epoch}.pt"
        torch.save(model.state_dict(), out_path)
        print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    # Expect: python -m src.training.train word transformer
    if len(sys.argv) != 3:
        print("Usage: python -m src.training.train <tokenizer_name> <model_type>")
        sys.exit(1)

    tok_name = sys.argv[1]
    model_type = sys.argv[2]
    train_model(tok_name, model_type)
