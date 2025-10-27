import sys, os, time, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import tokenizer
from models.xlstm import XLSTMLM

os.environ['KMP_DUPLICATE_LIB_OK']='True'

with open("config/hyperparameters_xlstm.json", "r") as f:
    cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

data_path = cfg.data_path
save_path = cfg.save_path
split_ratio = tuple(cfg.split_ratio)
block_size = cfg.block_size
batch_size = cfg.batch_size
patience = cfg.patience

d_model = cfg.d_model
n_layers = cfg.n_layers
dropout = cfg.dropout

learning_rate = cfg.learning_rate
weight_decay = cfg.weight_decay
grad_clip = cfg.grad_clip
max_iters = cfg.max_iters
eval_interval = cfg.eval_interval
eval_iters = cfg.eval_iters
seed = cfg.seed

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Dataset
if data_path == "data/tiny_shakespeare.txt":
    if os.path.exists(data_path):
        print(f"'{data_path}' already exists, skipping download.")
    else:
        import requests
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        os.makedirs("data", exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Tiny Shakespeare downloaded! File size:", len(text), "characters")
elif data_path == "data/full_shakespeare.txt":
    if os.path.exists(data_path):
        print(f"'{data_path}' already exists, skipping download.")
    else:
        import requests
        os.makedirs("data", exist_ok=True)
        url = "https://www.gutenberg.org/files/100/100-0.txt"
        print("Downloading full Shakespeare from Project Gutenberg...")
        text = requests.get(url).text
        if "*** START" in text:
            text = text.split("*** START")[1]
        if "*** END" in text:
            text = text.split("*** END")[0]
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Full Shakespeare downloaded! File size:", len(text), "characters")
else:
    print("Unexpected dataset, stop training.")
    sys.exit()

# Tokenizer
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()
tok = tokenizer.CharTokenizer(text)
print(len(tok.chars), "unique chars")

ids = tok.encode(text)
data = torch.tensor(ids, dtype=torch.long)

vocab_size = getattr(tok, "vocab_size", len(tok.chars))
print("vocab_size =", vocab_size)
mx = int(max(ids)) if len(ids) > 0 else -1
assert mx < int(vocab_size), f"max id {mx} >= vocab_size {int(vocab_size)}"

n = len(data)
n_train = int(split_ratio[0] * n)
n_val = int(split_ratio[1] * n)
n_test = n - n_train - n_val

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f"Total tokens: {n:,}")
print(f"Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")

def get_batch(split):
    data_split = {"train": train_data, "val": val_data, "test": test_data}[split]
    ix = torch.randint(0, len(data_split) - block_size - 1, (batch_size,))
    x = torch.stack([data_split[i     : i + block_size]     for i in ix])
    y = torch.stack([data_split[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print(xb.shape, yb.shape)

# Model
class MiniXLSTMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = XLSTMLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, block_size=block_size, dropout=dropout)

    def forward(self, idx, targets=None):
        logits, loss, _ = self.lm(idx, targets=targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None):
        return self.lm.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

model = MiniXLSTMLM().to(device)
print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

run_dir = f"runs/xlstm_{time.strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir=run_dir)
print("TensorBoard logdir:", run_dir)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

best_val   = float("inf")
bad_epochs = 0

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

@torch.no_grad()
def estimate_loss(split):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(split)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

for step in range(1, max_iters + 1):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    writer.add_scalar("loss/train", loss.item(), step)

    if step % eval_interval == 0 or step == 1:
        val_loss = estimate_loss("val")
        print(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")
        scheduler.step(val_loss)
        writer.add_scalar("loss/val", val_loss, step)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_val": best_val,
            }, save_path)
            print(f"improved! best_val={best_val:.4f} (saved)")
        else:
            bad_epochs += 1
            print(f"no improvement ({bad_epochs}/{patience})")
            if bad_epochs >= patience:
                print("early stopping triggered.")
                break

@torch.no_grad()
def evaluate_test_set():
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch("test")
        _, loss = model(xb, yb)
        losses.append(loss.item())
    test_loss = sum(losses) / len(losses)
    return test_loss

checkpoint = torch.load(save_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device).eval()
print(f"Loaded best model with best_val={checkpoint['best_val']:.4f} at step={checkpoint['step']}")

with torch.no_grad():
    test_loss = evaluate_test_set()

print(f"Final Test Loss: {test_loss:.4f}")
print(f"Perplexity (PPL): {torch.exp(torch.tensor(test_loss)):.2f}")

model.eval()
start_ids = tok.encode("ROMEO:")
idx = torch.tensor([start_ids], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=400, temperature=0.9, top_k=50)
print(tok.decode(out[0].tolist()))
