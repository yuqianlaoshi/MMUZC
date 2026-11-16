"""Usage: python exablation.py"""
import os
from datetime import datetime

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import cohen_kappa_score, f1_score
from scipy import stats


class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))


class FusionLoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, aux_dim, proj_dim, rank=16):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.A = nn.Linear(proj_dim + aux_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x1, x2):
        return self.base(x1) + self.B(self.A(torch.cat([self.proj(x1), x2], dim=1)))


class Late_Fusion_MLP(nn.Module):
    def __init__(self, in_x, in_x2, hidden, n_classes, num_fusions=3):
        super().__init__()
        self.num_fusions = num_fusions
        self.linear1 = nn.Linear(in_x, hidden)
        self.linear2 = FusionLoRALinear(hidden, 256, aux_dim=in_x2, proj_dim=64)
        self.linear3 = FusionLoRALinear(256, 128, aux_dim=in_x2, proj_dim=32)
        self.Relu = TeLU()
        self.out = FusionLoRALinear(128, n_classes, aux_dim=in_x2, proj_dim=16)

    def forward(self, x, x2):
        zero = torch.zeros(x.size(0), x2.size(1), device=x.device, dtype=x.dtype)
        x2_l2 = x2 if self.num_fusions >= 3 else zero
        x2_l3 = x2 if self.num_fusions >= 2 else zero
        h = self.Relu(self.linear1(x))
        h = self.Relu(self.linear2(h, x2_l2))
        h = self.Relu(self.linear3(h, x2_l3))
        return self.out(h, x2)


def get_model(name, in_x, in_x2, hidden, n_classes):
    nf = {'lf_fuse1': 1, 'lf_fuse2': 2, 'late_fusion': 3}[name]
    return Late_Fusion_MLP(in_x, in_x2, hidden, n_classes, num_fusions=nf)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal.mean()
        return focal.sum()


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

code_map = {'A': 0, 'S': 1, 'R': 2, 'G': 3, 'B': 4, 'U': 5, 'M': 6}
label_names = ['A', 'S', 'R', 'G', 'B', 'U', 'M']

df = pd.read_excel(os.path.join(DATA_DIR, 'label.xlsx'),
                   sheet_name=0, header=None)
rows = df.values.tolist()
y = []
cnt = 0
for i in range(1, 8897):
    if cnt < len(rows) and i > rows[cnt][0]:
        cnt += 1
    y.append(code_map[rows[cnt][2]])

df = pd.read_csv(os.path.join(DATA_DIR, 'image_embeddings04271_swinT.csv'))
df['Image Name'] = df['Image Name'].str.extract(r'(\d+)').astype(int)
df_sorted = df.sort_values(by='Image Name').reset_index(drop=True)
X = df_sorted.iloc[:, 1:].values

df = pd.read_csv(os.path.join(DATA_DIR, 'output.csv'), header=None)
rows = df.values.tolist()
row = [r[:19] for r in rows]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
x1_all = torch.tensor(row, dtype=torch.float32)

indices = np.arange(len(y_tensor))[::5]
X_tensor = X_tensor[indices]
y_tensor = y_tensor[indices]
x1_tensor = x1_all[indices]

print(f'samples: {len(y_tensor)}')

SEEDS = list(range(42, 52))
hidden_dim = 512
bs = 1024
ABLATION_HP = {'lr': 0.005, 'epochs': 110}
ABLATION_MODELS = ['lf_fuse1', 'lf_fuse2', 'late_fusion']
REF_MODEL = 'late_fusion'
ALPHA = 0.05

OUT_FILE = f'ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'


def wilcoxon_p(a, b):
    d = np.array(a) - np.array(b)
    nz = d[d != 0]
    if len(nz) == 0:
        return float('nan'), 0
    _, p = stats.wilcoxon(nz)
    return float(p), len(nz)


def nb_corrected_t(a, b, n_test, n_train):
    d = np.array(a) - np.array(b)
    m = len(d)
    se = d.std(ddof=1) * np.sqrt(1.0 / m + n_test / n_train)
    t = d.mean() / se
    p = float(2 * stats.t.sf(abs(t), df=m - 1))
    return t, p


def holm_adjusted(pvals):
    p = np.asarray(pvals, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        adj[i] = min(1.0, max(prev, p[i] * (m - rank)))
        prev = adj[i]
    return adj


def ms_str(vals):
    if not vals:
        return '-'
    if len(vals) == 1:
        return f'{vals[0]:.3f}'
    return f'{np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}'


def run_ablation(model_name, seed):
    set_seed(seed)

    dataset = TensorDataset(X_tensor, x1_tensor, y_tensor)
    n_train = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [n_train, len(dataset) - n_train])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = get_model(model_name,
                      in_x=X_tensor.shape[1], in_x2=x1_tensor.shape[1],
                      hidden=hidden_dim, n_classes=7)
    loss_func = FocalLoss(alpha=1, gamma=3)
    optimizer = optim.Adam(model.parameters(), lr=ABLATION_HP['lr'])

    for epoch in range(ABLATION_HP['epochs']):
        model.train()
        for Xb, x1b, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_func(model(Xb, x1b), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'  [{model_name} seed={seed}] epoch {epoch+1}/{ABLATION_HP["epochs"]}  loss={loss.item():.4f}')

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, x1b, yb in test_loader:
            out = model(Xb, x1b)
            _, pred = torch.max(out, 1)
            y_true.append(yb.cpu())
            y_pred.append(pred.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    acc = (y_pred == y_true).mean()
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cls_acc = {label_names[c]: float((y_pred[y_true == c] == c).mean())
               for c in range(7) if (y_true == c).any()}
    return acc, f1, kappa, cls_acc


n_train = int(0.8 * len(y_tensor))
n_test = len(y_tensor) - n_train

print(f'ablation ({len(SEEDS)} seeds 8:2 random split, seeds={SEEDS[0]}~{SEEDS[-1]}, lr={ABLATION_HP["lr"]}, epochs={ABLATION_HP["epochs"]})')
print(f'train/test: {n_train}/{n_test}')

results = {name: {'acc': [], 'f1': [], 'kappa': [], 'cls': {c: [] for c in label_names}}
           for name in ABLATION_MODELS}

for seed in SEEDS:
    for name in ABLATION_MODELS:
        acc, f1, kappa, cls_acc = run_ablation(name, seed)
        r = results[name]
        r['acc'].append(acc)
        r['f1'].append(f1)
        r['kappa'].append(kappa)
        for c, v in cls_acc.items():
            r['cls'][c].append(v)
        print(f'  [seed {seed}] {name:<12} Acc={acc:.4f}  F1={f1:.4f}  Kappa={kappa:.4f}')

with open(OUT_FILE, 'w') as f:
    def emit(s):
        print(s)
        f.write(s + '\n')

    emit(f'ablation ({len(SEEDS)} seeds 8:2 random split, seeds={SEEDS[0]}~{SEEDS[-1]}, lr={ABLATION_HP["lr"]}, epochs={ABLATION_HP["epochs"]})')
    emit(f'train/test: {n_train}/{n_test}')
    emit('')

    for name in ABLATION_MODELS:
        r = results[name]
        n_fuse = 3 if name == 'late_fusion' else int(name[-1])
        emit('=' * 70)
        emit(f'{name} (num_fusions={n_fuse})')
        emit(f'{"Seed":<8}{"Acc":<12}{"F1-macro":<12}{"Kappa":<12}')
        for seed, acc, f1, kappa in zip(SEEDS, r['acc'], r['f1'], r['kappa']):
            emit(f'{seed:<8}{acc:<12.4f}{f1:<12.4f}{kappa:<12.4f}')
        emit('-' * 70)
        emit(f'{"Mean":<8}{np.mean(r["acc"]):<12.4f}{np.mean(r["f1"]):<12.4f}{np.mean(r["kappa"]):<12.4f}')
        emit(f'{"Std":<8}{np.std(r["acc"], ddof=1):<12.4f}{np.std(r["f1"], ddof=1):<12.4f}{np.std(r["kappa"], ddof=1):<12.4f}')
        emit('  per-class Acc (Mean±Std): ' + '  '.join(f'{c}:{ms_str(r["cls"][c])}' for c in label_names))
        emit('')

    others = [n for n in ABLATION_MODELS if n != REF_MODEL]
    if len(SEEDS) >= 6 and others:
        emit('=' * 70)
        emit(f'paired significance tests (ref: {REF_MODEL}; {len(SEEDS)} paired runs)')
        emit('tests: paired t / Nadeau-Bengio corrected t / Wilcoxon')
        emit(f'conclusion by Holm-adjusted Wilcoxon p (alpha={ALPHA}); diff = ref - variant')
        emit('')

        wil_p = {}
        for m in others:
            for key in ['acc', 'f1', 'kappa']:
                wil_p[(m, key)] = wilcoxon_p(results[REF_MODEL][key], results[m][key])
        holm_map = dict(zip(wil_p.keys(), holm_adjusted([v[0] for v in wil_p.values()])))

        for key, mname in zip(['acc', 'f1', 'kappa'], ['Acc', 'F1-macro', 'Kappa']):
            emit(f'--- {mname} ---')
            emit(f'{"variant":<14}{"diff mean±std":<20}{"wins":<10}{"p(paired-t)":<12}{"p(NB)":<12}{"p(Wilcoxon)":<14}{"p(Holm)":<12}{"verdict"}')
            for m in others:
                a = np.array(results[REF_MODEL][key])
                b = np.array(results[m][key])
                d = a - b
                wins = int((d > 0).sum())
                _, p_t = stats.ttest_rel(a, b)
                _, p_nb = nb_corrected_t(a, b, n_test, n_train)
                p_w, nz = wil_p[(m, key)]
                p_h = holm_map[(m, key)]
                verdict = 'significant' if (not np.isnan(p_h)) and p_h < ALPHA else 'n.s.'
                diff_str = f'{d.mean():.4f} ± {d.std(ddof=1):.4f}'
                emit(f'{m:<14}{diff_str:<20}{f"{wins}/{len(SEEDS)}":<10}'
                     f'{p_t:<12.4f}{p_nb:<12.4f}{p_w:<14.4f}{p_h:<12.4f}{verdict}')
            emit('')
        if any(v[1] < 6 for v in wil_p.values()):
            emit('note: some Wilcoxon comparisons have < 6 nonzero diffs.')
    else:
        emit('fewer than 6 seeds, significance tests skipped.')

print(f'\nresults saved to {OUT_FILE}')
