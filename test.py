from src.tokenization.factory import get_tokenizer

tok = get_tokenizer("bpe")

ids = tok.encode("Hello   world!! This is   test.")
print("IDs:", ids)
print("Decoded:", tok.decode(ids))