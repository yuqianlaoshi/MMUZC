import torch
import torch.nn as nn
import torch.nn.functional as F

# ── base ──

class TeLU(nn.Module):
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
        return self.base(x) + self.B(self.A(x))


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
        x1_proj = self.proj(x1)
        fused = torch.cat([x1_proj, x2], dim=1)
        lora_out = self.B(self.A(fused))
        return base_out + lora_out


# ── late fusion (main) ──

class Late_Fusion_MLP(nn.Module):
    def __init__(self, in_x, in_x2, hidden, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_x, hidden)
        self.linear2 = FusionLoRALinear(hidden, 128, aux_dim=in_x2, proj_dim=64)
        self.linear3 = FusionLoRALinear(128, 64, aux_dim=in_x2, proj_dim=32)
        self.Relu = TeLU()
        self.out = FusionLoRALinear(64, n_classes, aux_dim=in_x2, proj_dim=16)

    def forward(self, x, x2):
        h = self.linear1(x)
        h = self.Relu(h)
        h = self.linear2(h, x2)
        h = self.Relu(h)
        h = self.linear3(h, x2)
        h = self.Relu(h)
        return self.out(h, x2)


class BaseMLP(nn.Module):
    def __init__(self, in_x, in_x2, hidden, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_x + in_x2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x, x2):
        return self.mlp(torch.cat([x, x2], dim=1))


class GatedMLP(nn.Module):
    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10, dropout=0.5):
        super().__init__()

        self.x_encoder = nn.Sequential(
            nn.Linear(in_x, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.x2_encoder = nn.Sequential(
            nn.Linear(in_x2, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden)
        self.out = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )

    def forward(self, x, x2):
        h_x = self.x_encoder(x)
        h_x2 = self.x2_encoder(x2)
        g = self.gate_network(torch.cat([h_x, h_x2], dim=1))
        fused = g * h_x + (1 - g) * h_x2
        fused = self.fusion_norm(fused)
        return self.out(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10,
                 num_heads=4, dropout=0.3):
        super().__init__()

        self.x_encoder = nn.Sequential(
            nn.Linear(in_x, hidden), TeLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.x2_encoder = nn.Sequential(
            nn.Linear(in_x2, hidden), TeLU(),
            nn.Linear(hidden, hidden),
        )

        self.cross_x_to_x2 = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_x2_to_x = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden), TeLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), TeLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x, x2):
        h_x = self.x_encoder(x).unsqueeze(1)
        h_x2 = self.x2_encoder(x2).unsqueeze(1)

        attended_x, _ = self.cross_x_to_x2(h_x, h_x2, h_x2)
        attended_x = self.norm1(h_x + attended_x)

        attended_x2, _ = self.cross_x2_to_x(h_x2, h_x, h_x)
        attended_x2 = self.norm2(h_x2 + attended_x2)

        fused = torch.cat([attended_x.squeeze(1), attended_x2.squeeze(1)], dim=1)
        return self.classifier(fused)


class MultimodalTransformer(nn.Module):
    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10,
                 num_layers=1, num_heads=2, dropout=0.3):
        super().__init__()

        self.x_proj = nn.Sequential(
            nn.Linear(in_x, hidden), TeLU(),
            nn.Linear(hidden, hidden),
        )
        self.x2_proj = nn.Sequential(
            nn.Linear(in_x2, hidden), TeLU(),
            nn.Linear(hidden, hidden),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.modal_emb = nn.Parameter(torch.randn(1, 2, hidden) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, 3, hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), TeLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x, x2):
        B = x.size(0)

        h_x = self.x_proj(x).unsqueeze(1)
        h_x2 = self.x2_proj(x2).unsqueeze(1)

        h_x = h_x + self.modal_emb[:, 0:1, :]
        h_x2 = h_x2 + self.modal_emb[:, 1:2, :]

        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, h_x, h_x2], dim=1)
        tokens = tokens + self.pos_emb

        out = self.transformer(tokens)
        out = self.norm(out[:, 0, :])
        return self.classifier(out)


class EarlyFusionMLP(nn.Module):
    def __init__(self, in_x=512, in_x2=19, hidden=256, n_classes=10, dropout=0.3):
        super().__init__()
        self.in_x2 = in_x2
        self.linear1 = FusionLoRALinear(in_x, hidden, aux_dim=in_x2, proj_dim=64)
        self.linear2 = FusionLoRALinear(hidden, hidden // 2, aux_dim=in_x2, proj_dim=32)
        self.linear3 = FusionLoRALinear(hidden // 2, n_classes, aux_dim=in_x2, proj_dim=16)
        self.act = TeLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x2):
        zero_x2 = torch.zeros(x.size(0), self.in_x2, device=x.device, dtype=x.dtype)

        h = self.linear1(x, x2)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h, zero_x2)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear3(h, zero_x2)
        return h


class MSFM_Fusion(nn.Module):
    def __init__(self, in_x, in_x2, hidden, n_classes):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_x + in_x2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.proj_rs = nn.Linear(in_x, hidden)
        self.proj_sv = nn.Linear(in_x2, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes)
        )

    def forward(self, x, x2):
        cat = torch.cat([x, x2], dim=1)
        w = self.gate(cat)
        rs_feat = self.proj_rs(x)
        sv_feat = self.proj_sv(x2)
        fused = w * rs_feat + (1.0 - w) * sv_feat
        return self.classifier(fused)


class IPCAM_Fusion(nn.Module):
    def __init__(self, in_x, in_x2, hidden, n_classes, reduction=4):
        super().__init__()
        ip_hidden = max(in_x2 // reduction, 8)
        self.sv_se = nn.Sequential(
            nn.Linear(in_x2, ip_hidden),
            nn.ReLU(),
            nn.Linear(ip_hidden, in_x2),
            nn.Sigmoid()
        )
        self.proj_rs = nn.Linear(in_x, hidden)
        self.proj_sv = nn.Linear(in_x2, hidden)
        concat_dim = hidden * 2
        self.se_fc1 = nn.Linear(concat_dim, concat_dim // reduction)
        self.se_fc2 = nn.Linear(concat_dim // reduction, concat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x, x2):
        sv_weight = self.sv_se(x2)
        sv_enhanced = x2 * sv_weight
        f_rs = self.proj_rs(x)
        f_sv = self.proj_sv(sv_enhanced)
        concat = torch.cat([f_rs, f_sv], dim=1)
        se_weight = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(concat))))
        gated = concat * se_weight
        return self.classifier(gated)


# ── registry ──

MODEL_REGISTRY = {
    'late_fusion':     Late_Fusion_MLP,
    'base':            BaseMLP,
    'gated':           GatedMLP,
    'cross_attn':      CrossAttentionFusion,
    'mm_transformer':  MultimodalTransformer,
    'early_fusion':    EarlyFusionMLP,
    'msfm':            MSFM_Fusion,
    'ipcam':           IPCAM_Fusion,
}


def get_model(name, in_x, in_x2, hidden, n_classes):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](in_x=in_x, in_x2=in_x2, hidden=hidden, n_classes=n_classes)
