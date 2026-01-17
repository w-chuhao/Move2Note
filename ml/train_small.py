from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = np.load(ROOT / "data" / "npz" / "train_windows.npz")
X = DATA["X"]  # [N,T,D]
y = DATA["y"]  # [N]

label_map = np.load(ROOT / "data" / "label_map.npy", allow_pickle=True).item()
num_classes = len(label_map)

T = X.shape[1]
D = X.shape[2]

# Stratified split: ensure each class is represented in train & val
from collections import defaultdict
N = X.shape[0]

# Group indices by class
class_indices = defaultdict(list)
for i, label in enumerate(y):
    class_indices[int(label)].append(i)

train_idx, val_idx = [], []
for cls, indices in class_indices.items():
    np.random.shuffle(indices)
    split_pt = max(int(len(indices) * 0.8), 1)
    train_idx.extend(indices[:split_pt])
    val_idx.extend(indices[split_pt:] if split_pt < len(indices) else indices[:1])

np.random.shuffle(train_idx)
np.random.shuffle(val_idx)

Xtr, Xva = X[train_idx], X[val_idx]
ytr, yva = y[train_idx], y[val_idx]
print(f"Train: {len(Xtr)}, Val: {len(Xva)}, Classes: {num_classes}")

Xtr = torch.from_numpy(Xtr)
ytr = torch.from_numpy(ytr)
Xva = torch.from_numpy(Xva) if len(Xva) else None
yva = torch.from_numpy(yva) if len(yva) else None


class GRUClassifier(nn.Module):
    def __init__(self, d_in, hidden, num_classes):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[0])


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRUClassifier(D, 128, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 regularization

# Class weights to handle imbalance
class_counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
crit = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))

ROOT.joinpath("models").mkdir(exist_ok=True)
best = -1.0
patience, no_improve = 10, 0  # Early stopping

for epoch in range(80):  # More epochs, early stop will kick in
    model.train()
    perm = torch.randperm(len(Xtr))

    for i in range(0, len(Xtr), 32):
        idx = perm[i:i+32]
        xb = Xtr[idx].to(device)
        yb = ytr[idx].to(device)

        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_logits = model(Xtr.to(device))
        tr_acc = (tr_logits.argmax(1).cpu() == ytr).float().mean().item()

        if Xva is not None and len(Xva) > 0:
            va_logits = model(Xva.to(device))
            va_acc = (va_logits.argmax(1).cpu() == yva).float().mean().item()
        else:
            va_acc = tr_acc  # no val set available

    if va_acc > best:
        best = va_acc
        no_improve = 0
        torch.save(model.state_dict(), ROOT / "models" / "exercise.pt")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stop at epoch {epoch}")
            break

    print(f"epoch {epoch:02d}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  best={best:.3f}")

print("saved:", ROOT / "models" / "exercise.pt")
