import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm


class BaseMLP(nn.Module):
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

def generate_data(n_samples, in_x, in_x2, n_classes):
    X = torch.randn(n_samples, in_x)
    X2 = torch.randn(n_samples, in_x2)
    Y = torch.randint(0, n_classes, (n_samples,))
    return X, X2, Y

class TeLU(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.B(self.A(x))
        return base_out + lora_out

class Late_Fusion_MLP(nn.Module):
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


class GatedMLP(nn.Module):
    """
    纯门控融合：没有交叉注意力，只有门控加权
    结构：x 和 x2 各自过MLP -> 门控融合 -> 输出
    """

    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10, dropout=0.5):
        super().__init__()

        # x 分支：深层MLP
        self.x_encoder = nn.Sequential(
            nn.Linear(in_x, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )

        # x2 分支：浅层MLP（19维不需要太复杂）
        self.x2_encoder = nn.Sequential(
            nn.Linear(in_x2, hidden // 2),  # 可以小一点
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden)
        )

        # 门控网络：根据两个特征生成融合权重
        self.gate_network = nn.Sequential(
            nn.Linear(hidden * 2, hidden),  # 输入是拼接后的特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Sigmoid()  # 输出 0~1 的权重
        )

        # 输出层
        self.fusion_norm = nn.LayerNorm(hidden)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x, x2):
        # 各自编码
        h_x = self.x_encoder(x)  # (B, hidden)
        h_x2 = self.x2_encoder(x2)  # (B, hidden)
        gate_input = torch.cat([h_x, h_x2], dim=1)  # (B, hidden*2)
        g = self.gate_network(gate_input)  # (B, hidden)，每个维度独立门控
        fused = g * h_x + (1 - g) * h_x2  # (B, hidden)

        # 归一化 + 输出
        fused = self.fusion_norm(fused)
        return self.out(fused)


class CMA_Block_1D(nn.Module):
    """
    CMA Block 适配 1D 向量输入
    原始：2D图像 (B,C,H,W) -> 1D特征 (B, L)
    """

    def __init__(self, in_x=512, in_x2=19, hidden=256, out_dim=256):
        super().__init__()
        self.hidden = hidden
        self.scale = hidden ** -0.5

        # Q 来自 x (512维)
        self.proj_q = nn.Sequential(
            nn.Linear(in_x, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )

        # K, V 来自 x2 (19维)
        self.proj_k = nn.Sequential(
            nn.Linear(in_x2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )
        self.proj_v = nn.Sequential(
            nn.Linear(in_x2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )

        # 输出投影 + 残差
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 如果维度不匹配，需要投影 x
        self.x_proj = nn.Linear(in_x, out_dim) if in_x != out_dim else nn.Identity()

    def forward(self, x, x2):
        """
        x: (B, 512)  - 作为主模态，提供 Query
        x2: (B, 19)  - 作为辅助模态，提供 Key/Value
        """
        B = x.size(0)

        # 生成 Q, K, V
        q = self.proj_q(x).unsqueeze(1)  # (B, 1, hidden) - 单token查询
        k = self.proj_k(x2).unsqueeze(1)  # (B, 1, hidden) - 单token键
        v = self.proj_v(x2).unsqueeze(1)  # (B, 1, hidden) - 单token值

        # 注意力：q 查询 k/v
        # 扩展为多头形式（这里用单头简化）
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, 1, 1)
        attn = attn.softmax(dim=-1)

        # 加权聚合
        out = torch.matmul(attn, v)  # (B, 1, hidden)
        out = out.squeeze(1)  # (B, hidden)

        # 投影 + 残差连接（x 经过投影后残差）
        residual = self.x_proj(x)
        out = self.out_proj(out)

        # 如果维度匹配直接加，否则只返回 out
        if out.shape == residual.shape:
            return residual + out
        else:
            return out


class CMA_FusionMLP(nn.Module):
    """
    基于 CMA 的融合网络：多层 CMA + 输出
    无门控，无复杂交叉，纯 CMA 结构堆叠
    """

    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10, num_layers=2):
        super().__init__()

        # 多层 CMA
        self.cma_layers = nn.ModuleList([
            CMA_Block_1D(
                in_x if i == 0 else hidden,
                in_x2,
                hidden,
                hidden
            ) for i in range(num_layers)
        ])

        # 最终输出
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x, x2):
        h = x
        for cma in self.cma_layers:
            h = cma(h, x2)  # 每层都用 x2 作为 K/V
        return self.out(h)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CMA_Block_1D(nn.Module):
    """
    CMA Block 适配 1D 向量输入
    原始：2D图像 (B,C,H,W) -> 1D特征 (B, L)
    """

    def __init__(self, in_x=512, in_x2=19, hidden=256, out_dim=256):
        super().__init__()
        self.hidden = hidden
        self.scale = hidden ** -0.5

        # Q 来自 x (512维)
        self.proj_q = nn.Sequential(
            nn.Linear(in_x, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )

        # K, V 来自 x2 (19维)
        self.proj_k = nn.Sequential(
            nn.Linear(in_x2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )
        self.proj_v = nn.Sequential(
            nn.Linear(in_x2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )

        # 输出投影 + 残差
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 如果维度不匹配，需要投影 x
        self.x_proj = nn.Linear(in_x, out_dim) if in_x != out_dim else nn.Identity()

    def forward(self, x, x2):
        """
        x: (B, 512)  - 作为主模态，提供 Query
        x2: (B, 19)  - 作为辅助模态，提供 Key/Value
        """
        B = x.size(0)

        # 生成 Q, K, V
        q = self.proj_q(x).unsqueeze(1)  # (B, 1, hidden) - 单token查询
        k = self.proj_k(x2).unsqueeze(1)  # (B, 1, hidden) - 单token键
        v = self.proj_v(x2).unsqueeze(1)  # (B, 1, hidden) - 单token值

        # 注意力：q 查询 k/v
        # 扩展为多头形式（这里用单头简化）
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, 1, 1)
        attn = attn.softmax(dim=-1)

        # 加权聚合
        out = torch.matmul(attn, v)  # (B, 1, hidden)
        out = out.squeeze(1)  # (B, hidden)

        # 投影 + 残差连接（x 经过投影后残差）
        residual = self.x_proj(x)
        out = self.out_proj(out)

        # 如果维度匹配直接加，否则只返回 out
        if out.shape == residual.shape:
            return residual + out
        else:
            return out


class CMA_FusionMLP(nn.Module):
    """
    基于 CMA 的融合网络：多层 CMA + 输出
    无门控，无复杂交叉，纯 CMA 结构堆叠
    """

    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10, num_layers=2):
        super().__init__()

        # 多层 CMA
        self.cma_layers = nn.ModuleList([
            CMA_Block_1D(
                in_x if i == 0 else hidden,
                in_x2,
                hidden,
                hidden
            ) for i in range(num_layers)
        ])

        # 最终输出
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x, x2):
        h = x
        for cma in self.cma_layers:
            h = cma(h, x2)  # 每层都用 x2 作为 K/V
        return self.out(h)


class SpectralFusionMLP(nn.Module):
    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10):
        super().__init__()

        self.spatial_branch = nn.Sequential(
            nn.Linear(in_x, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2)
        )

        # 频域分支：先投影到较小维度再做FFT，避免维度爆炸
        self.freq_proj = nn.Sequential(nn.Linear(in_x, hidden // 2), nn.ReLU())
        # FFT后：hidden//2 -> hidden//4 + 1 (实数FFT)，所以输入维度是 hidden//2
        self.freq_encoder = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),  # 修正：FFT输出维度
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2)
        )

        self.x2_branch = nn.Sequential(
            nn.Linear(in_x2, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2)
        )

        self.fusion = nn.Sequential(
            nn.Linear((hidden // 2) * 3, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x, x2):
        spatial = self.spatial_branch(x)  # (B, 128)

        # 频域
        h_fft = self.freq_proj(x)  # (B, 128)
        fft_out = torch.fft.rfft(h_fft, dim=1)  # (B, 65) - 128//2 + 1

        # 对齐维度：填充或截断到 hidden//2 = 128
        freq_dim = h_fft.size(1)  # 128
        fft_real = torch.real(fft_out)  # (B, 65)
        fft_imag = torch.imag(fft_out)

        # 拼接实部和虚部，然后投影到 freq_dim
        fft_cat = torch.cat([fft_real, fft_imag], dim=1)  # (B, 130)
        # 线性插值到目标维度
        fft_cat = F.interpolate(fft_cat.unsqueeze(1), size=freq_dim, mode='linear').squeeze(1)

        freq = self.freq_encoder(fft_cat)  # (B, 128)

        x2_feat = self.x2_branch(x2)  # (B, 128)

        print(f"  Spatial: {spatial.shape}, Freq: {freq.shape}, x2: {x2_feat.shape}")
        combined = torch.cat([spatial, freq, x2_feat], dim=1)  # (B, 384)
        return self.fusion(combined)