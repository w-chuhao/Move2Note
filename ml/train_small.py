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

# Tiny dataset: use simple split (last 20%) just to show training works
N = X.shape[0]
split = max(int(N * 0.8), 1)

Xtr, Xva = X[:split], X[split:]
ytr, yva = y[:split], y[split:]

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
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

ROOT.joinpath("models").mkdir(exist_ok=True)
best = -1.0

for epoch in range(40):
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
        torch.save(model.state_dict(), ROOT / "models" / "exercise.pt")

    print(f"epoch {epoch:02d}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  best={best:.3f}")

print("saved:", ROOT / "models" / "exercise.pt")
