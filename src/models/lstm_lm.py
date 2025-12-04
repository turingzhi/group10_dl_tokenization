import torch
import torch.nn as nn

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, model_dim, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.lstm = nn.LSTM(model_dim, model_dim, batch_first=True)
        self.fc = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.fc(out)
