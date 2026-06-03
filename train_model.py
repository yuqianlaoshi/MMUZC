import os
from datetime import datetime

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from scipy import stats as scipy_stats
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
from exmodel import get_model

import torch.nn.functional as F

from matplotlib import pyplot as plt

# ── loss ──

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
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


code_map = {'A': 0, 'S': 1, 'R': 2, 'G': 3, 'B': 4, 'U': 5, 'M': 6}

df = pd.read_excel('label.xlsx', sheet_name=0, header=None)
rows = df.values.tolist()
y = []
cnt = 0
for i in range(1, 8897):
    if cnt < len(rows) and i > rows[cnt][0]:
        cnt += 1
    y.append(code_map[rows[cnt][2]])

df = pd.read_csv('image_embeddings04271_swinT.csv')
df['Image Name'] = df['Image Name'].str.extract(r'(\d+)').astype(int)
df_sorted = df.sort_values(by='Image Name').reset_index(drop=True)
X = df_sorted.iloc[:, 1:].values

df = pd.read_csv('output.csv', header=None)
rows = df.values.tolist()
row = [r[:19] for r in rows]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
x1_tenor = torch.tensor(row, dtype=torch.float32)
indices = np.arange(len(y_tensor))[::5]

X_tensor = X_tensor[indices]
y_tensor = y_tensor[indices]
x1_tensor = x1_tenor[indices]

cls_num = torch.zeros(7, dtype=torch.long)
for c in range(7):
    cls_num[c] = (y_tensor == c).sum().item()
print(f'Class distribution: {cls_num.tolist()}')
print(f'Label mapping: {code_map}\n')

MODEL_NAME = 'late_fusion'
hidden_dim = 256
learning_rate = 0.0005
num_epochs = 100
bs = 1024

N_RUNS = 10
BASE_SEED = 42
ALPHA = 0.05


def run_experiment(seed):
    set_seed(seed)

    dataset = TensorDataset(X_tensor, x1_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    input_dim = X_tensor.shape[1]
    input_fusion_dim = x1_tensor.shape[1]
    output_dim = 7

    model = get_model(MODEL_NAME,
        in_x=input_dim, in_x2=input_fusion_dim,
        hidden=hidden_dim, n_classes=output_dim
    )

    loss_func = FocalLoss(alpha=1, gamma=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        for Xb, x1b, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_func(model(Xb, x1b), yb)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(model(Xb, x1b), dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total
        if (epoch + 1) % 50 == 0:
            print(f'  [seed={seed}] Epoch {epoch+1:>3}/{num_epochs}, '
                  f'Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}')

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
    return {
        'seed': seed,
        'accuracy': acc,
        'f1_macro': f1,
        'kappa': kappa,
    }


def bootstrap_ci(data, n_bootstrap=10000, alpha=0.05):
    data = np.array(data)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper


def one_sample_ttest(values, chance_level=1/7):
    t_stat, p_val = scipy_stats.ttest_1samp(values, chance_level)
    return t_stat, p_val


OUT_DIR = f'results_{MODEL_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(OUT_DIR, exist_ok=True)
print(f'Model: {MODEL_NAME}')
print(f'Output dir: {OUT_DIR}\n')

print(f'{N_RUNS} runs (seeds: {BASE_SEED} ~ {BASE_SEED + N_RUNS - 1})')
print('=' * 60)

results = []
for run_idx, seed in enumerate(range(BASE_SEED, BASE_SEED + N_RUNS)):
    print(f'\n>>> Run {run_idx + 1}/{N_RUNS} (seed={seed})')
    r = run_experiment(seed)
    results.append(r)
    print(f'  Acc={r["accuracy"]:.4f}  F1={r["f1_macro"]:.4f}  Kappa={r["kappa"]:.4f}')

accs = [r['accuracy'] for r in results]
f1s = [r['f1_macro'] for r in results]
kappas = [r['kappa'] for r in results]

acc_ci = bootstrap_ci(accs, alpha=ALPHA)
f1_ci = bootstrap_ci(f1s, alpha=ALPHA)
kappa_ci = bootstrap_ci(kappas, alpha=ALPHA)

acc_ttest_t, acc_ttest_p = one_sample_ttest(accs)
f1_ttest_t, f1_ttest_p = one_sample_ttest(f1s)
kappa_ttest_t, kappa_ttest_p = one_sample_ttest(kappas)

print('\n' + '=' * 60)
print(f'              Results Summary -- {MODEL_NAME}')
print('=' * 60)
print(f'{"Metric":<16} {"Mean +/- Std":<20} {"95% CI":<28} {"p-value (vs chance)":<20}')
print('-' * 60)

for name, vals, ci, t, p in [
    ('Accuracy', accs, acc_ci, acc_ttest_t, acc_ttest_p),
    ('F1-macro', f1s, f1_ci, f1_ttest_t, f1_ttest_p),
    ('Kappa', kappas, kappa_ci, kappa_ttest_t, kappa_ttest_p),
]:
    mean, std = np.mean(vals), np.std(vals, ddof=1)
    sig = ' ***' if p < 0.001 else ' **' if p < 0.01 else ' *' if p < 0.05 else ''
    print(f'{name:<16} {mean:.4f} +/- {std:.4f}    '
          f'[{ci[0]:.4f}, {ci[1]:.4f}]    '
          f'p = {p:.2e}{sig}')

print('-' * 60)
print('* p < 0.05  |  ** p < 0.01  |  *** p < 0.001')

fig, axes = plt.subplots(1, 3, figsize=(10, 4), dpi=130)
for ax, vals, title in zip(axes,
                           [accs, f1s, kappas],
                           ['Accuracy', 'F1-macro', "Cohen's Kappa"]):
    ax.boxplot(vals)
    ax.set_title(title)
    ax.set_ylabel('Value')
    ax.set_xticklabels([])
    ax.scatter([1] * len(vals), vals, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/{MODEL_NAME}_boxplot.png', dpi=130)
plt.close()

with open(f'{OUT_DIR}/{MODEL_NAME}_results.txt', 'w') as f:
    f.write(f'Model: {MODEL_NAME}\n')
    f.write(f'Results ({N_RUNS} runs, seeds {BASE_SEED}~{BASE_SEED + N_RUNS - 1})\n')
    f.write('=' * 60 + '\n')
    f.write('Per-run:\n')
    for r in results:
        f.write(f'  seed={r["seed"]:>3}:  Acc={r["accuracy"]:.4f}  '
                f'F1={r["f1_macro"]:.4f}  Kappa={r["kappa"]:.4f}\n')
    f.write('\nSummary Mean +/- Std:\n')
    f.write(f'  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs, ddof=1):.4f}\n')
    f.write(f'  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1):.4f}\n')
    f.write(f'  Kappa:    {np.mean(kappas):.4f} +/- {np.std(kappas, ddof=1):.4f}\n')
    f.write(f'\n95% Bootstrap CI:\n')
    f.write(f'  Accuracy: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]\n')
    f.write(f'  F1-macro: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]\n')
    f.write(f'  Kappa:    [{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]\n')
    f.write(f'\nOne-sample t-test (H0: mean = 1/7 ~ 0.1429):\n')
    f.write(f'  Accuracy: t={acc_ttest_t:.3f}, p={acc_ttest_p:.2e}\n')
    f.write(f'  F1-macro: t={f1_ttest_t:.3f}, p={f1_ttest_p:.2e}\n')
    f.write(f'  Kappa:    t={kappa_ttest_t:.3f}, p={kappa_ttest_p:.2e}\n')

print(f'\nAll results saved to {OUT_DIR}/')
