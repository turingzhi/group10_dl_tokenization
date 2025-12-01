import sys
import math
import torch
import torch.nn.functional as F

from src.config import config
from src.data.load_tinystories import load_tinystories
from src.data.load_wiki2 import load_wiki2
from src.data.make_dataloaders import make_dataloaders
from src.tokenization.factory import get_tokenizer
from src.models.transformer_lm import TransformerLM
from src.models.lstm_lm import LSTMLM


def build_model(tokenizer, model_type):
    vocab_size = tokenizer.vocab_size
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
    return model


def eval_val_loss(tokenizer_name, model_type, checkpoint_path, split): # TODO! Change to test when needed
    device = config["device"]
    tokenizer = get_tokenizer(tokenizer_name)

    # build model and load weights
    model = build_model(tokenizer, model_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # load data (same subset)
    train_ds, val_ds, test_ds = load_wiki2()
    #_, val_loader = make_dataloaders(train_ds, val_ds, tokenizer, config)
    #train_loader, val_loader = make_dataloaders(train_ds, val_ds, tokenizer, config)

    if split == "val":
        eval_ds = val_ds
    elif split == "test":
        eval_ds = test_ds
    elif split == "train": #just a test that there is no issue with training set eval
        eval_ds = train_ds
    else:
        raise ValueError(f"Unknown split: {split} (expected 'val' or 'test')")

    _, val_loader = make_dataloaders(train_ds, eval_ds, tokenizer, config)
    
    # count total characters in eval_ds
    total_chars = 0
    total_bytes = 0
    for ex in eval_ds:
        text = ex["text"]
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    avg_nll = total_loss / total_tokens          # nats per token
    ppl = math.exp(avg_nll)                      # perplexity

    nll_per_char = total_loss / total_chars            # nats per character
    bpc = nll_per_char / math.log(2.0)                 # bits per character

    tokens_per_char = total_tokens / total_chars
    bits_per_token = (total_loss / math.log(2.0)) / total_tokens

    nll_per_byte = total_loss / total_bytes
    bits_per_byte = nll_per_byte / math.log(2.0)
    #print(f"Validation: nll/token={avg_nll:.4f}, perplexity={ppl:.4f}")
    print(f"{split} split results for tokenizer='{tokenizer_name}', model='{model_type}':")
    print(f"  nll/token       = {avg_nll:.4f} nats")
    print(f"  perplexity      = {ppl:.4f}")
    print(f"  nll/char        = {nll_per_char:.4f} nats")
    print(f"  bits per char   = {bpc:.4f} bits")
    print(f"  nll/byte        = {nll_per_byte:.4f} nats")
    print(f"  bits per byte   = {bits_per_byte:.4f} bits")
    print(f"  tokens per char = {tokens_per_char:.4f}") # Compression Rate
    print(f"  bits per token  = {bits_per_token:.4f}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python -m src.evaluation.eval_model <tokenizer> <model_type> <checkpoint_path> <split>")
        sys.exit(1)

    tok_name = sys.argv[1]       # e.g. "word"
    model_type = sys.argv[2]     # "transformer" or "lstm"
    ckpt = sys.argv[3]
    split = sys.argv[4]          # "val", "test", or "train"

    eval_val_loss(tok_name, model_type, ckpt, split)


if __name__ == "__main__":
    main()
