import torch
import torch.nn as nn
import torch.nn.functional as F

class MILSTMCell(nn.Module):
    """
    Multiplicative-Integration LSTM (a simple, strong LSTM variant).
    This is NOT the full xLSTM from the paper, but a compact xLSTM-like cell
    that often performs better than vanilla LSTM on char-level language modeling.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Separate linear maps for input and hidden for MI gates
        self.W = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        # LayerNorm to stabilize training (xLSTM-like flavor)
        self.ln_gates = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

    def forward(self, x, state):
        h, c = state  # each (B, H)

        gates_x = self.W(x)        # (B, 4H)
        gates_h = self.U(h)        # (B, 4H)

        # multiplicative integration
        gates = gates_x + gates_h + gates_x * gates_h
        gates = self.ln_gates(gates)

        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        c = self.ln_c(c)
        h = o * torch.tanh(c)
        return h, (h, c)


class MILSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([
            MILSTMCell(input_size if l == 0 else hidden_size, hidden_size) for l in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        """
        x: (B, T, C_in)
        state: list of tuples (h, c) for each layer, each (B, H)
        returns:
            y: (B, T, H), new_state
        """
        B, T, C = x.shape
        if state is None:
            state = [
                (x.new_zeros(B, self.hidden_size), x.new_zeros(B, self.hidden_size))
                for _ in range(self.num_layers)
            ]

        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            new_state = []
            for l, cell in enumerate(self.layers):
                h, c = state[l]
                h, (h, c) = cell(inp, (h, c))
                inp = h if l == self.num_layers - 1 else self.drop(h)  # dropout on inter-layer
                new_state.append((h, c))
            state = new_state
            outputs.append(inp)

        y = torch.stack(outputs, dim=1)  # (B, T, H)
        return y, state


class XLSTMLM(nn.Module):
    """Character LM on top of MILSTM backbone."""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # RNNs don't need positions; keep it simple.
        self.backbone = MILSTM(d_model, d_model, n_layers, dropout=dropout)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, state=None):
        # idx: (B, T)
        x = self.token_emb(idx)  # (B, T, d_model)
        y, state = self.backbone(x, state=state)  # (B, T, d_model)
        y = self.ln_f(y)
        logits = self.head(y)    # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss, state

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None):
        self.eval()
        B = idx.size(0)
        state = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            x = self.token_emb(idx_cond)  # (B, T, C)
            y, state = self.backbone(x, state=state)  # keep state for efficiency
            last = self.ln_f(y[:, -1, :])             # (B, C)
            logits = self.head(last) / max(1e-6, temperature)  # (B, V)
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
