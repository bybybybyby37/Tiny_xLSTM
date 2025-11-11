import sys, os, time, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

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
num_epochs = cfg.eval_interval  
stride_overlap_ratio = cfg.stride_overlap_ratio

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

class CharDataset(Dataset):
    def __init__(self, data_tensor, block_size, stride=1):
        self.data = data_tensor
        self.block_size = block_size
        self.stride = stride
        self.n_samples = (len(self.data) - self.block_size) // self.stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        data_idx = idx * self.stride
        

        if data_idx + self.block_size + 1 > len(self.data):
            pass

        x = self.data[data_idx : data_idx + self.block_size]
        y = self.data[data_idx + 1 : data_idx + self.block_size + 1]
        return x, y
# define the stride ratio    
my_stride = int (block_size * stride_overlap_ratio)

train_dataset = CharDataset(train_data, block_size, stride=my_stride)
val_dataset = CharDataset(val_data, block_size, stride=block_size)
test_dataset = CharDataset(test_data, block_size, stride=block_size)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
)

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

# boost the training
if torch.__version__.startswith("2"):
    print("Compiling model...")
    model = torch.compile(model) 
    
print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

run_dir = f"runs/xlstm_{time.strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir=run_dir)
print("TensorBoard logdir:", run_dir)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

best_val   = float("inf")
bad_epochs = 0

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

scaler = torch.cuda.amp.GradScaler()

@torch.no_grad()
def estimate_loss(loader): 
    model.eval()
    losses = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device) 
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) 

global_step = 0 

print(f"Starting training for {num_epochs} epochs...")

for epoch in range(1, num_epochs + 1):
    print(f"--- Epoch {epoch}/{num_epochs} ---")
    model.train()

  
    for i, (xb, yb) in enumerate(train_loader):
        global_step += 1
        xb, yb = xb.to(device), yb.to(device) 

        with torch.cuda.amp.autocast():
            _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if i % 100 == 0:
             print(f"Epoch {epoch} | Step {i}/{len(train_loader)} | Train Loss {loss.item():.4f}")
        writer.add_scalar("loss/train", loss.item(), global_step)

    # validation after each epoch
    print(f"Epoch {epoch} complete. Running validation...")
    val_loss = estimate_loss(val_loader) 
    
    print(f"Epoch {epoch} | val_loss {val_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")
    
    scheduler.step(val_loss) 
    
    writer.add_scalar("loss/val", val_loss, global_step)
    writer.add_scalar("epoch", epoch, global_step)
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step)

    if val_loss < best_val:
        best_val = val_loss
        bad_epochs = 0
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch, 
            "best_val": best_val,
        }, save_path)
        print(f"improved! best_val={best_val:.4f} (saved)")
    else:
        bad_epochs += 1
        print(f"no improvement ({bad_epochs}/{patience})")
        if bad_epochs >= patience:
            print("early stopping triggered.")
            break 

# final test evaluation starts here
print("Training finished. Loading best model for final evaluation...")

@torch.no_grad()
def evaluate_test_set():
    model.eval()
    losses = []

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    test_loss = sum(losses) / len(losses)
    return test_loss

checkpoint = torch.load(save_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device).eval()
print(f"Loaded best model with best_val={checkpoint['best_val']:.4f} at epoch={checkpoint['epoch']}")

with torch.no_grad():
    test_loss = evaluate_test_set()

print(f"Final Test Loss: {test_loss:.4f}")
print(f"Perplexity (PPL): {torch.exp(torch.tensor(test_loss)):.2f}")

model.eval()
start_ids = tok.encode("ROMEO:")
idx = torch.tensor([start_ids], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=400, temperature=0.9, top_k=50)
print(tok.decode(out[0].tolist()))
