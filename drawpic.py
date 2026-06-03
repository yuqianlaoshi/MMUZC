import numpy as np
import matplotlib.pyplot as plt


def cross_entropy_loss(pt):
    return -np.log(pt)


def focal_loss(pt, alpha, gamma):
    return -alpha * (1 - pt) ** gamma * np.log(pt)


pt = np.linspace(0.01, 0.99, 100)
alpha = 1.0
gamma = 3.0

ce_loss = cross_entropy_loss(pt)
focal_loss_values = focal_loss(pt, alpha, gamma)

plt.figure(figsize=(10, 6))
plt.plot(pt, ce_loss, label='Cross Entropy Loss', color='blue')
plt.plot(pt, focal_loss_values, label=f'Focal Loss (alpha={alpha}, gamma={gamma})', color='red')

plt.title('Cross Entropy Loss vs Focal Loss')
plt.xlabel('Predicted Probability (pt)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()
