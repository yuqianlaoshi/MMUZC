import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    KAN 线性层：用 B-spline 基函数替代固定权重
    简化版：使用激活函数 + 可学习门控（类似 KAN 思想）
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 基础激活（类似 silu）
        self.base_activation = nn.SiLU()

        # 基础权重（残差连接）
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))

        # B-spline 网格参数
        self.grid = nn.Parameter(self._create_grid(grid_size, spline_order), requires_grad=False)

        # 可学习的 B-spline 系数
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))
        self.spline_bias = nn.Parameter(torch.zeros(out_features))

        self._init_weights()

    def _create_grid(self, grid_size, spline_order):
        """创建均匀网格"""
        grid = torch.linspace(-1, 1, grid_size + 1)
        grid = grid.unsqueeze(0).repeat(self.in_features, 1)
        # 扩展边界
        step = grid[:, 1:2] - grid[:, 0:1]
        for i in range(spline_order):
            grid = torch.cat([grid[:, :1] - step, grid, grid[:, -1:] + step], dim=1)
        return grid

    def _init_weights(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.xavier_uniform_(self.spline_weight)

    def forward(self, x):
        # 基础路径（类似 MLP）
        base_out = F.linear(self.base_activation(x), self.base_weight, self.base_bias)

        # B-spline 路径
        x_unsq = x.unsqueeze(-1)  # (B, in, 1)
        grid = self.grid.unsqueeze(0)  # (1, in, grid_size + 2*spline_order + 1)

        # 计算 B-spline 基函数
        bases = self._compute_bases(x_unsq, grid)  # (B, in, num_bases)

        # 加权求和
        spline_out = torch.einsum('bik,oik->bo', bases, self.spline_weight)
        spline_out = spline_out + self.spline_bias

        return base_out + spline_out

    def _compute_bases(self, x, grid):
        """计算 B-spline 基函数（De Boor 递归简化版）"""
        # 简化：使用线性插值近似（实际可用更复杂的递归）
        # 找到 x 所在的网格区间
        grid = grid.squeeze(0)  # (in, extended_grid)

        # 计算到每个网格点的距离，用高斯径向基近似 B-spline
        dist = (x - grid.unsqueeze(0)).pow(2)  # (B, in, grid_points)
        # 只取有效的基函数数量
        num_bases = self.grid_size + self.spline_order
        dist = dist[:, :, :num_bases]

        # 高斯核近似
        bases = torch.exp(-dist * (self.grid_size ** 2))
        bases = bases / (bases.sum(dim=-1, keepdim=True) + 1e-8)

        return bases


class KANMLP(nn.Module):
    """
    纯 KAN 结构：x 和 x2 各自过 KAN，然后拼接输出
    无交叉注意力，无门控
    """

    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10,
                 grid_size=5, spline_order=3):
        super().__init__()

        # x 分支：KAN 编码
        self.x_kan = nn.Sequential(
            KANLinear(in_x, hidden, grid_size, spline_order),
            KANLinear(hidden, hidden, grid_size, spline_order),
            KANLinear(hidden, hidden, grid_size, spline_order)
        )

        # x2 分支：简单 KAN（19维不需要太复杂）
        self.x2_kan = KANLinear(in_x2, hidden, grid_size, spline_order)

        # 输出：直接拼接后线性层（这里用普通 Linear 即可，也可用 KAN）
        self.out = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, x2):
        h_x = self.x_kan(x)  # (B, hidden)
        h_x2 = self.x2_kan(x2)  # (B, hidden)

        # 直接拼接，无门控，无交叉
        fused = torch.cat([h_x, h_x2], dim=1)  # (B, hidden*2)
        return self.out(fused)


# ============ 更高效的简化 KAN（推荐） ============

class EfficientKAN(nn.Module):
    """
    高效 KAN：使用分组卷积近似 B-spline，速度更快
    """

    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10,
                 num_basis=8):  # 基函数数量
        super().__init__()

        # 为每个输入维度学习一组基函数变换
        self.x_basis = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, num_basis), nn.SiLU(),
                nn.Linear(num_basis, hidden // in_x if i < in_x - 1 else hidden - (in_x - 1) * (hidden // in_x))
            ) for i in range(in_x)
        ])

        # 或者更简单的：径向基函数网络
        self.x_encoder = self._make_rbf_kan(in_x, hidden, num_basis)
        self.x2_encoder = self._make_rbf_kan(in_x2, hidden, num_basis)

        self.out = nn.Linear(hidden * 2, n_classes)

    def _make_rbf_kan(self, in_dim, out_dim, num_basis):
        """用 RBF 网络实现 KAN"""
        return nn.Sequential(
            # 第一层：RBF 激活
            RBFLayer(in_dim, num_basis),
            nn.Linear(num_basis, out_dim),
            nn.ReLU(),
            # 第二层
            RBFLayer(out_dim, num_basis),
            nn.Linear(num_basis, out_dim)
        )

    def forward(self, x, x2):
        h_x = self.x_encoder(x)
        h_x2 = self.x2_encoder(x2)
        return self.out(torch.cat([h_x, h_x2], dim=1))


class RBFLayer(nn.Module):
    """径向基函数层"""

    def __init__(self, in_features, num_basis):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_basis, in_features))
        self.widths = nn.Parameter(torch.ones(num_basis))

    def forward(self, x):
        # x: (B, in), centers: (num_basis, in)
        dist = torch.cdist(x, self.centers)  # (B, num_basis)
        return torch.exp(-(dist ** 2) / (2 * self.widths ** 2))