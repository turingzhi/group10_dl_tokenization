import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import math

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
        return config["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * progress))

def train_model(tokenizer_name: str, model_type: str, cfg_overrides=None):
    if cfg_overrides:
        config.update(cfg_overrides)
        print("new config")

    print(f"Starting training with tokenizer='{tokenizer_name}', model_type='{model_type}'")
    device = config["device"]

    # 1) tokenizer and data
    tokenizer = get_tokenizer(tokenizer_name)
    pad_id = tokenizer.pad_id
    print(f"pad:{pad_id}")

    print("Loading Wiki2...")
    train_raw, val_raw, test_raw = load_wiki2()
    # print("Loaded Wiki2:", len(train_raw), "train,", len(val_raw), "val", len(test_raw), "test")

    train_texts = [ex["text"] for ex in train_raw]
    val_texts   = [ex["text"] for ex in val_raw]
    test_texts  = [ex["text"] for ex in test_raw]

    print("Building tokenized datasets and dataloaders...")
    # NOTE: make_dataloaders must now also build test_loader, test_ds
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(train_raw, val_raw, test_raw, tokenizer, config)
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

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    results = []
    global_step = 0

    # 3) training loop
    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch} starting...")
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
                label_smoothing=0.1,
                ignore_index=pad_id,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

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
                    ignore_index=pad_id
                )
                val_loss += vloss.item()

        val_loss /= len(val_loader)

        # ---- compute val metrics ----
        nll_token = val_loss
        ppl = math.exp(nll_token)
        bits_per_token = nll_token / math.log(2)

        total_chars_val = sum(len(t) for t in val_texts)
        total_tokens_val = len(val_ds.ids)
        tokens_per_char = total_tokens_val / total_chars_val

        nll_per_char = (nll_token * total_tokens_val) / total_chars_val
        bpc = nll_per_char / math.log(2)

        total_bytes_val = sum(len(t.encode("utf-8")) for t in val_texts)
        total_nll_val = nll_token * total_tokens_val          
        nll_per_byte = total_nll_val / total_bytes_val        
        bits_per_byte = nll_per_byte / math.log(2.0)

        print(f"Epoch {epoch}:")
        print(f"  [VAL] nll/token       = {nll_token:.4f} nats")
        print(f"  [VAL] perplexity      = {ppl:.4f}")
        print(f"  [VAL] bits per token  = {bits_per_token:.4f}")
        print(f"  [VAL] nll/char        = {nll_per_char:.4f} nats")
        print(f"  [VAL] bits per char   = {bpc:.4f} bits")
        print(f"  [VAL] tokens per char = {tokens_per_char:.4f}")
        print(f"  [VAL] bits per byte   = {bits_per_byte:.4f}")

        results.append({
            "epoch": epoch,
            "split": "val",
            "nll_token": nll_token,
            "perplexity": ppl,
            "bits_per_token": bits_per_token,
            "nll_per_char": nll_per_char,
            "bits_per_char": bpc,
            "tokens_per_char": tokens_per_char,
            "bits_per_byte": bits_per_byte,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        })

        with open(f"results/metrics_{tokenizer_name}_{model_type}.json", "w") as f:
            json.dump(results, f, indent=2)

        out_path = f"checkpoints/{tokenizer_name}_{model_type}_epoch{epoch}.pt"
        torch.save(model.state_dict(), out_path)
        print(f"Saved checkpoint to {out_path}")

    # compute test metrics
    print("Evaluating on TEST set with final model...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tloss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=pad_id
            )
            test_loss += tloss.item()

    test_loss /= len(test_loader)

    # compute test metrics
    nll_token = test_loss
    ppl = math.exp(nll_token)
    bits_per_token = nll_token / math.log(2)

    total_chars_test = sum(len(t) for t in test_texts)
    total_tokens_test = len(test_ds.ids)
    tokens_per_char = total_tokens_test / total_chars_test

    nll_per_char = (nll_token * total_tokens_test) / total_chars_test
    bpc = nll_per_char / math.log(2)

    total_bytes_test = sum(len(t.encode("utf-8")) for t in test_texts)
    total_nll_test = nll_token * total_tokens_test        
    nll_per_byte = total_nll_test / total_bytes_test
    bits_per_byte = nll_per_byte / math.log(2.0)

    print("TEST metrics:")
    print(f"  [TEST] nll/token       = {nll_token:.4f} nats")
    print(f"  [TEST] perplexity      = {ppl:.4f}")
    print(f"  [TEST] bits per token  = {bits_per_token:.4f}")
    print(f"  [TEST] nll/char        = {nll_per_char:.4f} nats")
    print(f"  [TEST] bits per char   = {bpc:.4f} bits")
    print(f"  [TEST] tokens per char = {tokens_per_char:.4f}")
    print(f"  [TEST] bits per byte   = {bits_per_byte:.4f}")

    test_results = {
        "split": "test",
        "nll_token": nll_token,
        "perplexity": ppl,
        "bits_per_token": bits_per_token,
        "nll_per_char": nll_per_char,
        "bits_per_char": bpc,
        "tokens_per_char": tokens_per_char,
        "bits_per_byte": bits_per_byte,
    }
    with open(f"results/test_metrics_{tokenizer_name}_{model_type}.json", "w") as f:
        json.dump(test_results, f, indent=2)




if __name__ == "__main__":
    # Expect: python -m src.training.train word transformer
    if len(sys.argv) != 3:
        print("Usage: python -m src.training.train <tokenizer_name> <model_type>")
        sys.exit(1)

    tok_name = sys.argv[1]
    model_type = sys.argv[2]
    train_model(tok_name, model_type)
