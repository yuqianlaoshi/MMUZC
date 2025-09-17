import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# 定义模型
class LateFusionMLP(nn.Module):
    def __init__(self, in_x, in_x1, hidden, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_x, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)  # 第三层隐藏层
        )
        self.out = nn.Linear(hidden + in_x1, n_classes)

    def forward(self, x, x1):
        h = self.mlp(x)  # (B, hidden)
        h = torch.cat([h, x1], dim=1)  # (B, hidden + in_x1)
        return self.out(h)


# 生成模拟数据
def generate_data(n_samples, in_x, in_x1, n_classes):
    X = torch.randn(n_samples, in_x)
    X1 = torch.randn(n_samples, in_x1)
    Y = torch.randint(0, n_classes, (n_samples,))  # 确保 Y 是一维的
    return X, X1, Y


# 设置参数
n_samples = 1780
in_x = 512
in_x1 = 19
hidden = 256
n_classes = 10
batch_size = 64
epochs = 50

# 生成数据
X, X1, Y = generate_data(n_samples, in_x, in_x1, n_classes)


print(X.shape)
print(Y.shape)
print(X1.shape)

# 创建数据集
dataset = TensorDataset(X, X1, Y)

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))  # 80%的数据用于训练
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = LateFusionMLP(in_x, in_x1, hidden, n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 训练模型
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        x, x1, y = batch
        optimizer.zero_grad()
        outputs = model(x, x1)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_dataloader:
        x, x1, y = batch
        outputs = model(x, x1)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print(f'Accuracy of the network on the test set: {100 * correct / total:.2f}%')