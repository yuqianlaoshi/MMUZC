"""Usage: python test_zone.py <rs_file.csv|xlsx> [sv_file.csv|xlsx] [pt_file.pth] [--key col]"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix


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
        base_out = self.base(x1)
        fused = torch.cat([self.proj(x1), x2], dim=1)
        return base_out + self.B(self.A(fused))


class Late_Fusion_MLP(nn.Module):
    def __init__(self, in_x, in_x2, hidden, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_x, hidden)
        self.linear2 = FusionLoRALinear(hidden, 128, aux_dim=in_x2, proj_dim=64)
        self.linear3 = FusionLoRALinear(128, 64, aux_dim=in_x2, proj_dim=32)
        self.Relu = TeLU()
        self.out = FusionLoRALinear(64, n_classes, aux_dim=in_x2, proj_dim=16)

    def forward(self, x, x2):
        h = self.Relu(self.linear1(x))
        h = self.Relu(self.linear2(h, x2))
        h = self.Relu(self.linear3(h, x2))
        return self.out(h, x2)


SV_COLS = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle']

CODE2NAME = {0: 'A', 1: 'S', 2: 'R', 3: 'G', 4: 'B', 5: 'U', 6: 'M'}
NAME2CODE = {v: k for k, v in CODE2NAME.items()}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PT_PATH = None
key_col = None
data_args = []
i = 1
while i < len(sys.argv):
    a = sys.argv[i]
    if a == '--key':
        key_col = sys.argv[i + 1]
        i += 2
    elif a.endswith('.pth'):
        PT_PATH = a if os.path.exists(a) else PT_PATH
        i += 1
    else:
        data_args.append(a)
        i += 1

if len(data_args) < 1 or PT_PATH is None:
    print('Usage: python test_zone.py <rs_file.csv|xlsx> '
          '[sv_file.csv|xlsx] <pt_file.pth> [--key col]')
    sys.exit(1)

RS_FILE = data_args[0]
SV_FILE = data_args[1] if len(data_args) > 1 else None


def infer_dims(state_dict):
    in_x = state_dict['linear1.weight'].shape[1]
    hidden = state_dict['linear1.weight'].shape[0]
    proj_dim = state_dict['linear2.proj.weight'].shape[0]
    aux_dim = state_dict['linear2.A.weight'].shape[1] - proj_dim
    n_classes = state_dict['out.base.weight'].shape[0]
    return in_x, hidden, aux_dim, n_classes


def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    return pd.read_excel(path, sheet_name=0) if ext in ('.xlsx', '.xls') \
        else pd.read_csv(path)


state = torch.load(PT_PATH, map_location='cpu')
in_x, hidden, in_x2, n_classes = infer_dims(state)
print(f'Loaded: {PT_PATH}')
print(f'Model expects: RS dim={in_x}, SV dim={in_x2} (optional), '
      f'classes={n_classes}, hidden={hidden}')

model = Late_Fusion_MLP(in_x=in_x, in_x2=in_x2, hidden=hidden, n_classes=n_classes)
model.load_state_dict(state, strict=True)
model = model.to(DEVICE)
model.eval()

rs_df = read_file(RS_FILE)
rs_cols = [f'dim_{i}' for i in range(in_x)]
missing_rs = [c for c in rs_cols if c not in rs_df.columns]
if missing_rs:
    raise ValueError(
        f'Missing {len(missing_rs)} RS columns in {RS_FILE}, '
        f'e.g. {missing_rs[:5]}... Must contain dim_0..dim_{in_x-1}.')

has_label = 'label' in rs_df.columns
if has_label:
    labels = rs_df['label'].astype(str).str.strip()
    bad = set(labels.unique()) - set(NAME2CODE)
    if bad:
        raise ValueError(f'Unknown label value(s): {bad}. '
                         f'Allowed: {sorted(NAME2CODE.keys())}')

X_rs = rs_df[rs_cols].values.astype(np.float32)
print(f'RS file: {RS_FILE}  ({len(rs_df)} samples)')

X_sv = np.zeros((len(rs_df), in_x2), dtype=np.float32)
if SV_FILE is not None:
    sv_df = read_file(SV_FILE)
    missing_sv = [c for c in SV_COLS if c not in sv_df.columns]
    if missing_sv:
        raise ValueError(
            f'Missing street-view columns in {SV_FILE}: {missing_sv}.')
    if key_col is not None:
        if key_col not in rs_df.columns or key_col not in sv_df.columns:
            raise ValueError(f'Key column {key_col!r} not in both files.')
        sv_map = {k: r[SV_COLS].values.astype(np.float32)
                  for k, r in sv_df.set_index(key_col, drop=False).iterrows()}
        matched = 0
        for j, k in enumerate(rs_df[key_col].values):
            if k in sv_map:
                X_sv[j] = sv_map[k]
                matched += 1
        print(f'SV file: {SV_FILE}  matched by {key_col}: '
              f'{matched}/{len(rs_df)} (rest zero-filled)')
    else:
        if len(sv_df) != len(rs_df):
            raise ValueError(
                f'Row count mismatch: RS {len(rs_df)} vs SV {len(sv_df)}. '
                f'Use --key <col> or make them row-aligned.')
        X_sv = sv_df[SV_COLS].values.astype(np.float32)
        print(f'SV file: {SV_FILE}  ({len(sv_df)} rows, row-order aligned)')
else:
    print('No SV file given - running with zero street-view input.')

X_rs_t = torch.tensor(X_rs)
X_sv_t = torch.tensor(X_sv)

y_pred_all = []
with torch.no_grad():
    for i in range(0, len(X_rs_t), 1024):
        out = model(X_rs_t[i:i + 1024].to(DEVICE), X_sv_t[i:i + 1024].to(DEVICE))
        y_pred_all.append(torch.argmax(out, dim=1).cpu())
y_pred = torch.cat(y_pred_all).numpy()

out_df = rs_df.copy()
out_df['pred_class'] = [CODE2NAME[p] for p in y_pred]

if has_label:
    y_true = labels.map(NAME2CODE).values
    out_df['correct'] = (y_true == y_pred).astype(int)

    acc = (y_pred == y_true).mean()
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    print('\n' + '=' * 60)
    print('Functional-zone test results')
    print('=' * 60)
    print(f'Accuracy : {acc:.4f}')
    print(f'F1-macro : {f1:.4f}')
    print(f'Kappa    : {kappa:.4f}')
    print(f'Correct  : {(y_pred == y_true).sum()} / {len(y_true)}')
    print('\nPer-class accuracy:')
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == c).mean()
            print(f'  {CODE2NAME[c]}: n={mask.sum():>4}  acc={cls_acc:.4f}')
    print('\nConfusion matrix (rows=true, cols=pred):')
    print('      ' + ''.join(f'{CODE2NAME[c]:>6}' for c in range(n_classes)))
    for c in range(n_classes):
        print(f'  {CODE2NAME[c]}  ' + ''.join(f'{v:>6}' for v in cm[c]))
else:
    print('\nNo label column found - predictions only.')
    print(out_df['pred_class'].value_counts().to_string())

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
pred_file = os.path.join(SCRIPT_DIR, f'zone_pred_{ts}.csv')
out_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
print(f'\nPredictions saved: {pred_file}')
