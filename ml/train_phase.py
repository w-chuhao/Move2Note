from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = np.load(ROOT / "data" / "npz" / "phase_windows.npz")
X = DATA["X"]  # [N,T,D]
y = DATA["y"]  # [N,T]

label_map = np.load(ROOT / "data" / "phase_label_map.npy", allow_pickle=True).item()
num_classes = len(label_map)

T = X.shape[1]
D = X.shape[2]


class GRUPhase(nn.Module):
    def __init__(self, d_in, hidden, num_classes):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)  # [B,T,H]
        return self.fc(out)   # [B,T,C]


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRUPhase(D, 128, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss(ignore_index=-1)

Xtr = torch.from_numpy(X)
ytr = torch.from_numpy(y)

ROOT.joinpath("models").mkdir(exist_ok=True)

for epoch in range(40):
    model.train()
    perm = torch.randperm(len(Xtr))

    for i in range(0, len(Xtr), 16):
        idx = perm[i:i + 16]
        xb = Xtr[idx].to(device)
        yb = ytr[idx].to(device)

        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits.reshape(-1, num_classes), yb.reshape(-1))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xtr.to(device))
        pred = logits.argmax(-1).cpu()
        mask = ytr != -1
        acc = (pred[mask] == ytr[mask]).float().mean().item()
    print(f"epoch {epoch:02d} acc={acc:.3f}")

torch.save(model.state_dict(), ROOT / "models" / "phase.pt")
print("saved:", ROOT / "models" / "phase.pt")
