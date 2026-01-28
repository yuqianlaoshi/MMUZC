import numpy as np
import matplotlib.pyplot as plt

# 定义普通交叉熵损失函数
def cross_entropy_loss(pt):
    return -np.log(pt)

# 定义Focal Loss函数
def focal_loss(pt, alpha, gamma):
    return -alpha * (1 - pt) ** gamma * np.log(pt)

# 生成预测概率pt的范围，从0.01到0.99，足够密集以便绘图平滑
pt = np.linspace(0.01, 0.99, 100)
alpha = 1.0  # 类别平衡权重，可以根据需要调整
gamma = 3.0  # 调节易分类样本权重的参数，可以根据需要调整

# 计算损失值
ce_loss = cross_entropy_loss(pt)
focal_loss_values = focal_loss(pt, alpha, gamma)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(pt, ce_loss, label='Cross Entropy Loss', color='blue')
plt.plot(pt, focal_loss_values, label=f'Focal Loss (α={alpha}, γ={gamma})', color='red')

# 添加图例和标签
plt.title('Comparison of Cross Entropy Loss and Focal Loss')
plt.xlabel('Predicted Probability (pt)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()