import torch
import torch.nn as nn

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)                        # (B, T, d_model)
        pos = torch.arange(T, device=idx.device)              # (T,)
        pos_emb = self.pos_emb(pos)[None, :, :]               # (1, T, d_model)
        return tok_emb + pos_emb