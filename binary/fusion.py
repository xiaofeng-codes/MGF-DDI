import torch
import torch.nn as nn
import torch.nn.functional as F


class TriViewFusion(nn.Module):
    """
    三视图特征融合模块 (Tri-View Attention Fusion)
    用于融合 v_macro, v_micro, v_inter
    """

    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()

        # 1. 特征投影 (确保输入特征在同一特征空间，虽然输入通常维度一致，但这层增加非线性)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ) for _ in range(3)  # 对应 macro, micro, inter
        ])

        # 2. 跨视图自注意力 (Self-Attention)
        # 将三个视图视为序列长度为3的序列，捕捉视图间的依赖关系
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_att = nn.LayerNorm(hidden_dim)

        # 3. 动态门控融合 (Gating Mechanism)
        # 学习每个视图的权重分数
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # 输出3个权重
        )

        # 4. 最终输出处理
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, v_macro, v_micro, v_inter):
        """
        Args:
            v_macro: (Batch, input_dim)
            v_micro: (Batch, input_dim)
            v_inter: (Batch, input_dim)
        Returns:
            fused_vector: (Batch, hidden_dim)
        """
        # A. 独立投影
        # input_dim -> hidden_dim
        h_macro = self.projectors[0](v_macro)
        h_micro = self.projectors[1](v_micro)
        h_inter = self.projectors[2](v_inter)

        # Stack into sequence: (Batch, 3, hidden_dim)
        seq = torch.stack([h_macro, h_micro, h_inter], dim=1)

        # B. Self-Attention (视图间交互)
        # attn_out: (Batch, 3, hidden_dim)
        attn_out, _ = self.attention(seq, seq, seq)

        # Residual + Norm
        seq = self.norm_att(seq + attn_out)

        # C. 动态加权融合
        # Flatten for gating: (Batch, 3 * hidden_dim)
        flat_seq = seq.view(seq.size(0), -1)

        # 计算权重: (Batch, 3) -> softmax
        weights = F.softmax(self.gate_fc(flat_seq), dim=-1)

        # 扩展权重维度以便广播: (Batch, 3, 1)
        weights = weights.unsqueeze(-1)

        # 加权求和: sum( (Batch, 3, hidden_dim) * (Batch, 3, 1) ) -> (Batch, hidden_dim)
        fused = (seq * weights).sum(dim=1)

        return self.out_norm(fused), weights