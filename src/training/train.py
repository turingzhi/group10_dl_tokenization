import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import math
import json


from src.config import config
from src.data.load_tinystories import load_tinystories
from src.data.load_wiki2 import load_wiki2
from src.data.make_dataloaders import make_dataloaders
from src.tokenization.factory import get_tokenizer
from src.models.transformer_lm import TransformerLM
from src.models.lstm_lm import LSTMLM

def get_lr(step, warmup_steps, total_steps):
        if step < warmup_steps:
            return config["learning_rate"] * (step + 1) / warmup_steps
        
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return config["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * progress))



def train_model(tokenizer_name: str, model_type: str):
    print(f"Starting training with tokenizer='{tokenizer_name}', model_type='{model_type}'")
    device = config["device"]

    # 1) tokenizer and dataset
    tokenizer = get_tokenizer(tokenizer_name)
    pad_id = tokenizer.pad_id

    print("Loading Wiki2...")
    train_raw, val_raw, test_raw = load_wiki2()
    print("Loaded Wiki2:", len(train_raw), "train,", len(val_raw), "val,", len(test_raw), "test")



    train_texts = [ex["text"] for ex in train_raw]
    val_texts   = [ex["text"] for ex in val_raw]

    train_loader, val_loader, train_ds, val_ds = make_dataloaders(train_raw, val_raw, tokenizer, config)
    print("Dataloaders ready. Starting epochs...")

    total_steps = config["num_epochs"] * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    vocab_size = tokenizer.vocab_size

    # 2) model
    if model_type == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            model_dim=config["model_dim"],
            n_heads=config["num_heads"],
            n_layers=config["num_layers"],
            context_length=config["context_length"],
            dropout=config["dropout"],
        )
    elif model_type == "lstm":
        model = LSTMLM(vocab_size, config["model_dim"])
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)

    # results folder
    os.makedirs("results", exist_ok=True)
    results = []

    global_step = 0

    # 3) training loop
    for epoch in range(config["num_epochs"]):
        print(f"\n===== Epoch {epoch} =====")
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"{tokenizer_name}-{model_type} epoch {epoch}"):
            lr = get_lr(global_step, warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=pad_id,
                label_smoothing=config.get("label_smoothing", 0.0),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

        train_loss = total_loss / len(train_loader)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vloss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                    ignore_index=pad_id)
                val_loss += vloss.item()

        val_loss /= len(val_loader)

        # ---- compute metrics ----
        # 1. nll per token
        nll_token = val_loss                        # in nats

        # 2. perplexity
        ppl = math.exp(nll_token)

        # 3. bits per token
        bits_per_token = nll_token / math.log(2)

        # character counts for val set
        total_chars = sum(len(t) for t in val_texts)

        # total tokens in validation pass
        # total_tokens = len(val_loader) * config["batch_size"] * config["context_length"]
        total_tokens = len(val_ds.ids)

        tokens_per_char = total_tokens / total_chars

        # 4. nll per char
        nll_per_char = (nll_token * total_tokens) / total_chars

        # 5. bits per char
        bpc = nll_per_char / math.log(2)

        # ---- print metrics ----
        print(f"Epoch {epoch}:")
        print(f"  nll/token       = {nll_token:.4f} nats")
        print(f"  perplexity      = {ppl:.4f}")
        print(f"  bits per token  = {bits_per_token:.4f}")
        print(f"  nll/char        = {nll_per_char:.4f} nats")
        print(f"  bits per char   = {bpc:.4f} bits")
        print(f"  tokens per char = {tokens_per_char:.4f}")

        # ---- save JSON metrics ----
        results.append({
            "epoch": epoch,
            "nll_token": nll_token,
            "perplexity": ppl,
            "bits_per_token": bits_per_token,
            "nll_per_char": nll_per_char,
            "bits_per_char": bpc,
            "tokens_per_char": tokens_per_char,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        })

        with open(f"results/metrics_{tokenizer_name}_{model_type}.json", "w") as f:
            json.dump(results, f, indent=2)

        # ---- save checkpoint ----
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/{tokenizer_name}_{model_type}_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    # Expect: python -m src.training.train word transformer
    if len(sys.argv) != 3:
        print("Usage: python -m src.training.train <tokenizer_name> <model_type>")
        sys.exit(1)

    tok_name = sys.argv[1]
    model_type = sys.argv[2]
    train_model(tok_name, model_type)
