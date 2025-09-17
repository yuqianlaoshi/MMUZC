import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from kan import KANLayer
import matplotlib.pyplot as plt
from tqdm import tqdm


class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(in_dim, rank, bias=False)  # 降维
        self.B = nn.Linear(rank, out_dim, bias=False)  # 升维
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        # 输入: [batch, ..., in_dim]（任意形状的图像嵌入）
        base_out = self.base(x)
        # 低秩调整项（A降维后B升维）
        lora_out = self.B(self.A(x))
        return base_out + lora_out  # 残差融合

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



# 定义模型
class LateFusionMLP(nn.Module):
    def __init__(self, in_x, in_x2, hidden, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_x, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)  # 第三层隐藏层
        )
        self.out = nn.Linear(hidden + in_x2, n_classes)

    def forward(self, x, x2):
        h = self.mlp(x)  # (B, hidden)
        h = torch.cat([h, x2], dim=1)  # (B, hidden + in_x2)
        return self.out(h)


# 生成模拟数据
def generate_data(n_samples, in_x, in_x2, n_classes):
    X = torch.randn(n_samples, in_x)
    X2 = torch.randn(n_samples, in_x2)
    Y = torch.randint(0, n_classes, (n_samples,))
    return X, X2, Y

class TeLU(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

class LORAMLP(nn.Module):
    def __init__(self, in_x, in_x2, hidden, n_classes):
        super().__init__()
        self.linear1=LoRALinear(in_x, hidden)
        self.linear2=LoRALinear(hidden, 128)
        self.linear3=LoRALinear(128+in_x2, 64)
        self.Relu = TeLU()
        self.out = LoRALinear(64, n_classes)

    def forward(self, x, x2):
        h = self.linear1(x)  # (B, hidden)
        h = self.Relu(h)
        h = self.linear2(h)  # (B, hidden)
        h = self.Relu(h)
        h = torch.cat([h, x2], dim=1)  # (B, hidden + in_x2)
        h = self.linear3(h)
        h = self.Relu(h)
        return self.out(h)







