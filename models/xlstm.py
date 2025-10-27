import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Safe exponential function (prevents overflow)
# ---------------------------------------------------------
def exp_clamped(x, clamp=5.0):
    return torch.exp(torch.clamp(x, min=-clamp, max=clamp))

# ---------------------------------------------------------
# sLSTM branch (standard sigmoid input gate)
# ---------------------------------------------------------
class SLSTMCell(nn.Module):
    """
    Stable LSTM-like cell with Multiplicative Integration + LayerNorm.
    Uses a standard sigmoid input gate.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.inp = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hid = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.ln_g = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

    def forward(self, x, state):
        h, c = state
        gx = self.inp(x)
        gh = self.hid(h)

        # Multiplicative Integration: Wx + Uh + Wx*Uh
        gates = gx + gh + gx * gh
        gates = self.ln_g(gates)

        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        c = self.ln_c(c)
        h = o * torch.tanh(c)
        return h, (h, c)

# ---------------------------------------------------------
# mLSTM branch (exponential write gate)
# ---------------------------------------------------------
class MLSTMCell(nn.Module):
    """
    mLSTM with exponential gating for memory updates.
    This is the key mechanism from xLSTM:
    c_t = f * c_{t-1} + exp(i) * g
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.inp = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hid = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.ln_g = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

    def forward(self, x, state):
        h, c = state
        gx = self.inp(x)
        gh = self.hid(h)

        gates = gx + gh + gx * gh
        gates = self.ln_g(gates)

        i, f, g, o = gates.chunk(4, dim=-1)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Exponential memory write strength
        write = exp_clamped(i)
        c = f * c + write * g
        c = self.ln_c(c)
        h = o * torch.tanh(c)
        return h, (h, c)

# ---------------------------------------------------------
# xLSTM Block: parallel (sLSTM + mLSTM) + learnable fusion + residual
# ---------------------------------------------------------
class XLSTMBlock(nn.Module):
    """
    Each block:
    - PreNorm input
    - Run through sLSTM and mLSTM in parallel
    - Fuse their outputs using a learnable gate
    - Apply residual connection
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.s_cell = SLSTMCell(d_model, d_model)
        self.m_cell = MLSTMCell(d_model, d_model)
        self.prenorm = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_t, state):
        (hs, cs), (hm, cm) = state
        xt = self.prenorm(x_t)

        hs, (hs, cs) = self.s_cell(xt, (hs, cs))
        hm, (hm, cm) = self.m_cell(xt, (hm, cm))

        # Learnable fusion between s-stream and m-stream
        alpha = torch.sigmoid(self.fusion(torch.cat([hs, hm], dim=-1)))
        out = alpha * hs + (1 - alpha) * hm
        out = self.drop(out)

        # Residual
        y = x_t + out
        return y, ((hs, cs), (hm, cm))

    def init_state(self, batch, d_model, device):
        z = torch.zeros(batch, d_model, device=device)
        return ((z, z.clone()), (z.clone(), z.clone()))

# ---------------------------------------------------------
# Stacked xLSTM backbone
# ---------------------------------------------------------
class XLSTMBackbone(nn.Module):
    """
    Stacked xLSTM blocks, time-unrolled inside forward().
    """
    def __init__(self, d_model: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([XLSTMBlock(d_model, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, state=None):
        B, T, C = x.shape
        device = x.device

        if state is None:
            state = [layer.init_state(B, C, device) for layer in self.layers]

        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            new_state = []
            for l, layer in enumerate(self.layers):
                inp, st = layer(inp, state[l])
                new_state.append(st)
            state = new_state
            outputs.append(inp)

        y = torch.stack(outputs, dim=1)
        y = self.norm(y)
        return y, state

# ---------------------------------------------------------
# Full Language Model with xLSTM backbone
# ---------------------------------------------------------
class XLSTMLM(nn.Module):
    """
    Token embedding → xLSTM backbone → linear head → logits
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.backbone = XLSTMBackbone(d_model, n_layers, dropout)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, state=None):
        x = self.token_emb(idx)
        y, state = self.backbone(x, state=state)
        logits = self.head(y)

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
        state = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            x = self.token_emb(idx_cond)
            y, state = self.backbone(x, state=state)
            last = y[:, -1, :]
            logits = self.head(last) / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
