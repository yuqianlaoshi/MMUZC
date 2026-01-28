import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from timm.utils import random_seed
from torch.utils.data import DataLoader, TensorDataset
import random
from exmodel import BaseMLP, Late_Fusion_MLP, GatedMLP, CMA_FusionMLP,SpectralFusionMLP
from kan import KANMLP

import torch.nn.functional as F

from matplotlib import pyplot as plt
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
    torch.cuda.manual_seed_all(seed_value)  # 如果使用GPU
    # 以下两个设置是为了确保PyTorch的某些操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置一个固定的随机种子
set_seed(42)

y=[]
code_map = {
    'A': 0,
    'S': 1,
    'R': 2,
    'G': 3,
    'B': 4,
    'U': 5,
    'M': 6
}

# sheet_name=0 表示读第 1 个工作表，header=None 把第一行当普通数据读
df = pd.read_excel('label.xlsx', sheet_name=0, header=None)

# 转成纯 Python 的二维 list
rows = df.values.tolist()
cnt=0
for i in range(1,8897,1):
    if(i<=rows[cnt][0]):
        y.append(code_map[rows[cnt][2]])
    else:
        cnt+=1
        y.append(code_map[rows[cnt][2]])




file_path = 'image_embeddings04271_swinT.csv'

# 读取表格
df = pd.read_csv(file_path)

df['Image Name'] = df['Image Name'].str.extract('(\d+)').astype(int)
df_sorted = df.sort_values(by='Image Name')

df_sorted.reset_index(drop=True, inplace=True)
X = df_sorted.iloc[:, 1:].values

df=pd.read_csv('output.csv',header=None)
rows=df.values.tolist()
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
print(cls_num)
#数据部分

#超参数设置
hidden_dim = 256
learning_rate = 0.0005


dataset = TensorDataset(X_tensor, x1_tensor, y_tensor)
train_size = int(0.8 * len(dataset))  # 80%的数据用于训练
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
bs=1024
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)


input_dim = X_tensor.shape[1]
input_fusion_dim=x1_tensor.shape[1]
output_dim = len(torch.unique(y_tensor))  # 假设标签是0到n_class-1
print(output_dim)
#model = BaseMLP(in_x=input_dim, in_x2=input_fusion_dim, hidden=hidden_dim, n_classes=output_dim)

#model = Late_Fusion_MLP(in_x=input_dim, in_x2=input_fusion_dim, hidden=hidden_dim, n_classes=output_dim)

num_epochs = 70
#model =GatedMLP(in_x=input_dim, in_x2=input_fusion_dim, hidden=hidden_dim, n_classes=output_dim)
#model =KANMLP(in_x=input_dim, in_x2=input_fusion_dim, hidden=hidden_dim, n_classes=output_dim)
#model =CMA_FusionMLP(in_x=input_dim, in_x2=input_fusion_dim, hidden=hidden_dim, n_classes=output_dim)
model =SpectralFusionMLP(in_x=input_dim, in_x2=input_fusion_dim, hidden=hidden_dim, n_classes=output_dim)

# 定义损失函数和优化器
#loss_func= nn.CrossEntropyLoss()
loss_func=FocalLoss(alpha=1, gamma=3)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losspic=[]
accpic   = []
# 训练模型A
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for X_batch, x1_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch,x1_batch)
        #loss = criterion(outputs, y_batch)
        loss = loss_func(outputs, y_batch)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(correct / total)
    losspic.append(float(loss.item()))
    accpic.append(correct / total)

model.eval()
y_true, y_pred = [], []          # 1. 新增两行容器
with torch.no_grad():
    for X_batch, x1_batch, y_batch in test_dataloader:
        outputs = model(X_batch, x1_batch)
        _, predicted = torch.max(outputs, 1)   # 不需要 .data

        y_true.append(y_batch.cpu())           # 2. 收集真实标签
        y_pred.append(predicted.cpu())         #    收集预测标签

# 3. 拼成 1-D numpy
y_true = torch.cat(y_true).numpy()
y_pred = torch.cat(y_pred).numpy()

# 4. 计算指标
accuracy = (y_pred == y_true).mean()
kappa    = cohen_kappa_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f'Accuracy: {accuracy:.4f}')
print(f'F1-macro: {f1_macro:.4f}')
print(f'Kappa:    {kappa:.4f}')


# 保存模型
torch.save(model.state_dict(), 'mlp_model.pth')
plt.figure(figsize=(5,3), dpi=130)

# 左 y 轴：Loss
ax1 = plt.gca()
ax1.plot(losspic, color='tab:blue', lw=1.8, label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右 y 轴：Accuracy
ax2 = ax1.twinx()
ax2.plot(accpic, color='tab:red', lw=1.8, label='Accuracy')
ax2.set_ylabel('Accuracy (%)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 标题 + 保存
plt.title(f'hd={hidden_dim} lr={learning_rate} γ={3}')
plt.tight_layout()
plt.savefig('loss_acc.png', dpi=300)
plt.show()